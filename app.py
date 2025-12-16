from flask import Flask, request, render_template, abort, redirect, url_for, flash, send_file, jsonify, render_template_string
from sqlalchemy import create_engine, MetaData, Table, select, or_, func, text, and_, insert, update, delete
# Import the `cast` and `float` functions for potential vector column casting
from sqlalchemy import cast, Float
from bs4 import BeautifulSoup
from werkzeug.utils import secure_filename
from markupsafe import Markup  # <-- added to render highlights safely
import os, re, io, ssl, json, time, hashlib, math
import pandas as pd
from collections import defaultdict
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.header import Header


# ======================= FLASK & UPLOAD CONFIG =======================
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64MB upload cap
ALLOWED_UPLOADS = {".xlsx", ".xls", ".csv",".doc", ".docx" }

# ======================= DB CONFIG =======================
DB_USER = os.getenv("TIDB_USER") or os.getenv("MYSQL_USER", "root")
DB_PASSWORD = os.getenv("TIDB_PASSWORD") or os.getenv("MYSQL_PASSWORD", "592230")
DB_HOST = os.getenv("TIDB_HOST") or os.getenv("MYSQL_HOST", "127.0.0.1")
DB_PORT = str(os.getenv("TIDB_PORT") or os.getenv("MYSQL_PORT", "3306"))
DB_NAME = os.getenv("TIDB_DB") or os.getenv("MYSQL_DB", "mydb")
TABLE_NAME = "acer_details"
DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    "?charset=utf8mb4"
)
CONNECT_ARGS = {}
DB_USE_TLS = os.getenv("DB_USE_TLS")
if DB_USE_TLS is not None:
    use_tls = DB_USE_TLS.strip().lower() in ("1", "true", "yes")
else:
    use_tls = (DB_PORT == "4000") or ("tidbcloud.com" in (DB_HOST or "").lower())
if use_tls:
    try:
        print("[TLS] /etc/secrets contents:", os.listdir("/etc/secrets"))
    except Exception as _e:
        print("[TLS] Could not list /etc/secrets:", _e)
    env_ca = (os.getenv("TIDB_SSL_CA") or "").strip()
    candidates = [p for p in [env_ca,
                              "/etc/secrets/TIDB_SSL_CA",
                              "/etc/secrets/isrgrootx1.pem",
                              "/etc/secrets/ca.pem",
                              "/etc/ssl/certs/isrgrootx1.pem",
                              "/etc/ssl/certs/ca-certificates.crt"] if p]
    ssl_ca_path = next((p for p in candidates if os.path.isfile(p)), None)
    if ssl_ca_path:
        CONNECT_ARGS = {"ssl": {"ca": ssl_ca_path}}
        print(f"[DB] TLS enabled with CA: {ssl_ca_path}")
    else:
        try:
            ctx = ssl.create_default_context()
            CONNECT_ARGS = {"ssl": ctx}
            print("[DB] TLS enabled with system CA bundle (no explicit CA file found).")
        except Exception as e:
            raise RuntimeError(
                "TLS required but no CA file found and system bundle failed. "
                "Set env TIDB_SSL_CA to /etc/secrets/<your-secret-file> or add a Secret File."
            ) from e
else:
    print("[DB] TLS not required (DB_USE_TLS=0 or non-TiDB settings).")
ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=CONNECT_ARGS)
metadata = MetaData()
Acer = Table(TABLE_NAME, metadata, autoload_with=ENGINE)

# ======================= COLUMN SETS =======================
ALL_COLS = [
    "id","case_law_number","name_of_party","state","year","type_of_court",
    "industry_sector","subject_matter1","subject_matter2","section","rule",
    "case_name","date","basic_detail","summary_head_note","citation",
    "question_answered","full_case_law",
]
# Search surface (fast & focused; explicitly NOT scanning full_case_law)
SEARCHABLE_COLS = [
    "case_law_number","name_of_party","case_name","state",
    "section","rule","citation","industry_sector",
    "subject_matter1","subject_matter2",
    "basic_detail","question_answered",
    "summary_head_note",
    "type_of_court"
]

# --- explicit word-search fields per earlier ask ---
WORD_SEARCH_COLS = [
    "case_law_number", "name_of_party", "subject_matter1", "subject_matter2", "summary_head_note"
]

# ======================= OPENAI CONFIG =======================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_env_model = (os.getenv("OPENAI_MODEL") or "").strip()
_ALLOWED = {"gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"}
OPENAI_MODEL = _env_model if _env_model in _ALLOWED else "gpt-4o-mini"
# Embeddings (kept for other features; not used for AI search anymore)
EMBED_MODEL = (os.getenv("OPENAI_EMBED_MODEL") or "text-embedding-3-small").strip()
EMBED_COL_CANDIDATES = [
    "embedding","embeddings","text_embedding","text_embeddings",
    "summary_embedding","headnote_embedding","openai_embedding",
    "vector","vec","text_vector"
]
W_VEC   = float(os.getenv("HYBRID_W_VEC", "0.7"))
W_LEX   = float(os.getenv("HYBRID_W_LEX", "0.3"))
SCORE_MIN = float(os.getenv("HYBRID_SCORE_MIN", "0.18"))
CANDIDATE_PRIMARY = int(os.getenv("HYBRID_CANDIDATES_PRIMARY", "400"))
CANDIDATE_WIDE    = int(os.getenv("HYBRID_CANDIDATES_WIDE", "800"))
VECTOR_SWEEP_LIMIT = int(os.getenv("VECTOR_SWEEP_LIMIT", "1200"))
VECTOR_TOPK        = int(os.getenv("VECTOR_TOPK", "250"))
VEC_ONLY_MIN       = float(os.getenv("VEC_ONLY_MIN", "0.30"))

# Predefined admin users: email -> hashed_password
# Password: "admin123"
ADMIN_USERS = {
    "admin@acergst.com": "562b649530dccabdaaa91aeccbdbb218f19da0f13963b97442621a34beb876fc",  # sha256("admin123")
}

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def _openai_chat(messages, max_tokens=800, temperature=0.35):
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e_new:
        try:
            import openai  # type: ignore
            openai.api_key = OPENAI_API_KEY
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
            )
            return (resp["choices"][0]["message"]["content"] or "").strip()
        except Exception as e_old:
            raise RuntimeError(f"OpenAI call failed: {e_old or e_new}")

def _openai_chat_json(messages, model=None, max_tokens=900, temperature=0.3):
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
    model = model or OPENAI_MODEL or "gpt-4o-mini"
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=model, messages=messages,
            response_format={"type": "json_object"},
            max_tokens=max_tokens, temperature=temperature,
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception:
        import openai  # type: ignore
        openai.api_key = OPENAI_API_KEY
        _msgs = messages + [{"role": "system", "content": "Return a single JSON object as your entire reply."}]
        resp = openai.ChatCompletion.create(
            model=model, messages=_msgs,
            max_tokens=max_tokens, temperature=temperature,
        )
        raw = resp["choices"][0]["message"]["content"] or "{}"
        try:
            return json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, flags=re.S)
            return json.loads(m.group(0)) if m else {}

def _openai_embed(text: str, model: str = None):
    if not OPENAI_API_KEY:
        return None
    model = model or EMBED_MODEL
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.embeddings.create(model=model, input=text)
        vec = resp.data[0].embedding
        return vec
    except Exception as e_new:
        try:
            import openai  # type: ignore
            openai.api_key = OPENAI_API_KEY
            resp = openai.Embedding.create(model=model, input=text)
            return resp["data"][0]["embedding"]
        except Exception as e_old:
            print("Embedding error:", e_old or e_new)
            return None

@app.before_request
def require_login_for_admin_prefix():
    import time as _t
    path = (request.path or "")
    if not path.startswith("/admin"):
        return
    if 'admin_email' not in session:
        return redirect(url_for('login', next=request.url))
    now = int(_t.time())
    last_seen = session.get('last_seen_at')
    if last_seen and (now - last_seen) > 1800:
        session.pop('admin_email', None)
        session.pop('last_seen_at', None)
        return redirect(url_for('login', next=request.url))
    session['last_seen_at'] = now


from functools import wraps
from flask import Flask, session, redirect, url_for, request, render_template

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_email' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ======================= RUNTIME CACHES =======================
_AI_CACHE = {}
_META_CACHE = {}

def _ai_cache_get(key: str):
    e = _AI_CACHE.get(key)
    if not e: return None
    if e["exp"] < time.time():
        _AI_CACHE.pop(key, None)
        return None
    return e["value"]

def _ai_cache_set(key: str, value, ttl_sec: int = 900):
    _AI_CACHE[key] = {"value": value, "exp": time.time() + ttl_sec}

def _meta_cache_get(name: str):
    e = _META_CACHE.get(name)
    if not e: return None
    if e["exp"] < time.time():
        _META_CACHE.pop(name, None)
        return None
    return e["value"]

def _meta_cache_set(name: str, value, ttl_sec: int = 300):
    _META_CACHE[name] = {"value": value, "exp": time.time() + ttl_sec}

# ======================= SEARCH HELPERS =======================
def like_filters(q: str):
    q = (q or "").strip()
    if not q:
        return text("0=1")
    terms = []
    for col in SEARCHABLE_COLS:
        if hasattr(Acer.c, col):
            terms.append(getattr(Acer.c, col).like(f"%{q}%"))
    return or_(*terms) if terms else text("0=1")

# ---- Expanded stopwords to better ignore prepositions/generic words in fallback search ----
_STOPWORDS = {
    "a","an","the","of","and","or","in","on","to","by","for","with","under","about",
    "this","that","these","those","some","at","as","from","into","over","after","before",
    "between","through","during","without","within","above","below","up","down","out",
    "off","than","too","very","again","further","then","once"
}
_KEEP_SHORT = {"itc","gst","rcm","hsn","pos","b2b","b2c"}
_DOMAIN_PHRASES = [
    "intermediary services","intermediary service",
    "input tax credit","place of supply","reverse charge",
    "works contract","advance ruling",
    "e way bill","e-way bill","zero rated","zero-rated",
    "export of services","blocked credit","time of supply",
    "credit note","debit note","classification dispute",
    "valuation","anti profiteering","refund of itc",
    "job work","pure agent"
]

def _nlq_keywords(q: str):
    if not q:
        return []
    s = q.lower()
    s = re.sub(r"[^\w\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    kept, seen = [], set()
    s_for_phrase = " " + s + " "
    for ph in sorted(_DOMAIN_PHRASES, key=len, reverse=True):
        ph_norm = " " + ph + " "
        if ph_norm in s_for_phrase:
            if ph not in seen:
                kept.append(ph); seen.add(ph)
            s_for_phrase = s_for_phrase.replace(ph_norm, " ")
    residual = re.sub(r"\s+", " ", s_for_phrase).strip()
    if residual:
        for tok in residual.split():
            if tok in _STOPWORDS:
                continue
            if len(tok) <= 2 and tok not in _KEEP_SHORT:
                continue
            if tok not in seen:
                kept.append(tok); seen.add(tok)
    return kept

def _nlq_clause(query: str):
    toks = _nlq_keywords(query)
    if not toks:
        simple = " ".join(sorted(_tokenize(query)))
        return like_filters(simple if simple else query)
    phrases = [t for t in toks if " " in t]
    singles = [t for t in toks if " " not in t]
    must_clauses = [like_filters(p) for p in phrases] if phrases else []
    should_clause = or_(*[like_filters(t) for t in singles]) if singles else None
    if must_clauses and should_clause is not None:
        return and_(*must_clauses, should_clause)
    if must_clauses:
        return and_(*must_clauses)
    if should_clause is not None:
        return should_clause
    return like_filters(query)

def _merge_ai_and_user_keywords(ai_text: str, user_query: str) -> str:
    ai_tokens = _nlq_keywords(ai_text or "")
    ai_compact = " ".join(ai_tokens) if ai_tokens else ""
    user_query = (user_query or "").strip()
    if ai_compact and user_query:
        return f"{ai_compact} {user_query}"
    return ai_compact or user_query

def _build_semantic_query(ai_q: str, ai_text: str) -> str:
    toks = _nlq_keywords(ai_q)
    phrases = [t for t in toks if " " in t][:2]
    singles = [t for t in toks if " " not in t][:6]
    core = " ; ".join(phrases) if phrases else " ".join(singles)
    if not core:
        core = (ai_q or "").strip()
    return f"GST case law: {core}"

def _detect_embedding_column_name():
    env_name = (os.getenv("DB_EMBED_COL") or "").strip()
    if env_name and hasattr(Acer.c, env_name):
        print(f"[Vector] Using embedding column: {env_name} (from DB_EMBED_COL)")
        return env_name
    for nm in EMBED_COL_CANDIDATES:
        if hasattr(Acer.c, nm):
            print(f"[Vector] Using embedding column: {nm} (auto-detected)")
            return nm
    print("[Vector] No embedding column found. Set DB_EMBED_COL to your vector column name.")
    return None

def _parse_vec(val):
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        try:
            return [float(x) for x in val]
        except Exception:
            return None
    if isinstance(val, (bytes, bytearray)):
        try:
            val = val.decode("utf-8", "ignore")
        except Exception:
            val = str(val)
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [float(x) for x in arr]
            except Exception:
                pass
        try:
            parts = [p for p in re.split(r"[,\s]+", s) if p]
            if len(parts) >= 8:
                return [float(p) for p in parts]
        except Exception:
            return None
    return None

def _cosine(a, b):
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i]*b[i] for i in range(n))
    na = math.sqrt(sum(a[i]*a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i]*b[i] for i in range(n)))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def _tokenize(text):
    if not text: return set()
    text = text.lower()
    text = re.sub(r"[^\w\s\-]"," ", text)
    return set(t for t in text.split() if t and (len(t) > 2 or t in _KEEP_SHORT))

def _lexical_match_score(q_tokens, row):
    if not q_tokens:
        return 0.0
    weights = {
        "case_law_number": 2.0,
        "name_of_party": 1.5,
        "subject_matter1": 1.2,
        "subject_matter2": 1.2,
        "summary_head_note": 1.0,
    }
    fields = {}
    fields["case_law_number"] = (row.get("case_law_number") or "")
    fields["name_of_party"]   = (row.get("name_of_party") or "")
    fields["subject_matter1"] = (row.get("subject_matter1") or "")
    fields["subject_matter2"] = (row.get("subject_matter2") or "")
    fields["summary_head_note"] = _strip_all_html(row.get("summary_head_note") or "")
    tokenized = {k: _tokenize(v) for k, v in fields.items()}
    max_possible = sum(weights.values()) * len(q_tokens)
    if max_possible == 0:
        return 0.0
    score = 0.0
    for tok in q_tokens:
        for fld, toks in tokenized.items():
            if tok in toks:
                score += weights[fld]
    if fields["case_law_number"]:
        digits_in_case = re.findall(r"\d+", fields["case_law_number"])
        for d in digits_in_case[:3]:
            if d and d in " ".join(q_tokens):
                score += 0.6
                break
    return max(0.0, min(1.0, score / (max_possible + 1e-9)))

# ======================= TEXT CLEANERS =======================
def _strip_all_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = soup.get_text(separator="\n")
    text = re.sub(r"\r\n?","\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n", text)
    return text.strip()

def _bold_held_and_prefix(head_txt: str) -> str:
    if not head_txt:
        return ""
    def repl_held(m): return f"<strong>{m.group(0)}</strong>"
    head_txt = re.sub(r"\b(Held)\b", repl_held, head_txt, flags=re.IGNORECASE)
    lines = head_txt.split("\n")
    for i, line in enumerate(lines):
        if line.strip():
            idx = line.find("-")
            if idx > 0:
                prefix = line[:idx].strip()
                rest = line[idx:]
                lines[i] = f"<strong>{prefix}</strong>{rest}"
            break
    html = "<br>".join([re.sub(r" {2,}", " ", l) for l in lines])
    return html

def _split_points(text_block: str):
    if not text_block:
        return []
    txt = text_block.strip()
    pattern = re.compile(r"(?:(?<=\n)|^)(?:\d+\.\s+|[a-zA-Z]\)\s+|\([a-zA-Z]\)\s+)")
    parts = pattern.split(txt)
    if len(parts) > 1:
        items = []
        indices = [m.start() for m in pattern.finditer(txt)] + [len(txt)]
        if indices and indices[0] != 0:
            indices = [0] + indices
        for a, b in zip(indices, indices[1:]):
            chunk = txt[a:b].strip()
            chunk = re.sub(r"^(?:\d+\.|[a-zA-Z]\)|\([a-zA-Z]\))\s*", "", chunk).strip()
            if chunk:
                items.append(chunk)
        if items:
            return items
    return [ln.strip() for ln in re.split(r"\n{2,}|-\s+", txt) if ln.strip()]

# --------- Summary helpers ----------
def _selected_topic(args) -> str:
    s1 = args.getlist("s1")
    s2 = args.getlist("s2")
    sections = args.getlist("section")
    rules = args.getlist("rule")
    inds = args.getlist("industry")
    is_ai = args.get("ai_search") == "true"
    ai_q = (args.get("ai_q") or "").strip() if is_ai else ""
    q = (args.get("q") or "").strip()
    if s2:   return ", ".join(s2)
    if s1:   return ", ".join(s1)
    if sections: return "Section(s): " + ", ".join(sections)
    if rules:    return "Rule(s): " + ", ".join(rules)
    if inds:     return "Industry: " + ", ".join(inds)
    if ai_q:     return ai_q
    if q:        return q
    return "GST case law"

def _clean_summary(text: str) -> str:
    if not text: return ""
    lines = []
    for ln in text.splitlines():
        ln = re.sub(r'^\s*[\*\-\u2022\u00a0\u00a2]+\s*', '', ln)
        ln = re.sub(r'^\s*\d+\.\s*', '', ln)
        lines.append(ln)
    text = "\n".join(lines)
    text = re.sub(r'(?im)^\s*(overall|in summary|in conclusion|to conclude|in short)\b.*$', '', text)
    text = re.sub(r'\n{3,}', '\n', text).strip()
    return text

# ======================= AI EXPLANATION HELPERS (kept as-is for compatibility) =======================
def get_ai_explanation(prompt_text):
    # NOTE: Intentionally returning empty strings so UI remains stable
    # while AI search is disengaged per your instruction.
    return {
        "statute_text": "",
        "circular_text": "",
        "disclaimer": "It is advisable for the user to refer to the latest provisions of the law.",
        "cached": False
    }

# --- Kept for compatibility (not used for searching now) ---
def _get_common_cols():
    return [
        Acer.c.id, Acer.c.case_law_number, Acer.c.name_of_party, Acer.c.date,
        Acer.c.summary_head_note,
        Acer.c.type_of_court, Acer.c.state, Acer.c.year,
        Acer.c.section, Acer.c.rule, Acer.c.citation,
        Acer.c.subject_matter1, Acer.c.subject_matter2,
    ]

def _run_query(expr, add_cols=None, limit=200, offset=0):
    cols = _get_common_cols()
    if add_cols:
        cols += add_cols
    with ENGINE.connect() as conn:
        rows = conn.execute(
            select(*cols).where(expr).order_by(Acer.c.id.desc()).limit(limit).offset(offset)
        ).mappings().all()
    return [dict(r) for r in rows]

# ======================= NEW: Plain word-search utilities (phrase-first) =======================
def _normalize_phrase(q: str) -> str:
    """Normalize a phrase for LIKE by collapsing whitespace; keep original order/casing aside from spaces."""
    s = (q or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _word_terms(q: str):
    """
    Tokenize to words for matching/highlighting; drop tiny stopwords except domain short keeps.
    Also include the full normalized phrase as the first highlight term so phrase matches are emphasized.
    """
    if not q: return []
    phrase = _normalize_phrase(q).lower()
    s = re.sub(r"[^\w\s\-]", " ", phrase)
    toks = [t for t in s.split() if t and (len(t) > 2 or t in _KEEP_SHORT) and t not in _STOPWORDS]
    # de-duplicate while preserving order
    seen = set()
    out = []
    if phrase and len(phrase.split()) >= 1:
        out.append(phrase)  # full phrase first
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def _phrase_like_clause(q: str):
    """OR across WORD_SEARCH_COLS for the full phrase; fail closed on empty."""
    phrase = _normalize_phrase(q)
    if not phrase:
        return text("0=1")
    parts = []
    for col in WORD_SEARCH_COLS:
        if hasattr(Acer.c, col):
            parts.append(getattr(Acer.c, col).like(f"%{phrase}%"))
    return or_(*parts) if parts else text("0=1")

def _word_like_clause(q: str):
    """OR across WORD_SEARCH_COLS for ANY of the terms; fail closed on empty."""
    terms = _word_terms(q)
    tokens_only = [t for t in terms if " " not in t]
    if not tokens_only:
        return text("0=1")
    or_parts = []
    for term in tokens_only:
        for col in WORD_SEARCH_COLS:
            if hasattr(Acer.c, col):
                or_parts.append(getattr(Acer.c, col).like(f"%{term}%"))
    return or_(*or_parts) if or_parts else text("0=1")

def _highlight_text(html_or_text: str, terms: list[str]) -> str:
    """
    Phrase-first highlighting that avoids nested <mark> tags and respects existing HTML.
    - Wraps the full phrase first (if provided).
    - Temporarily protects already highlighted segments while highlighting single-word tokens.
    - Uses word-boundary highlighting for single terms.
    """
    if not html_or_text or not terms:
        return html_or_text or ""

    s = str(html_or_text)

    # ensure unique terms, longest first
    uniq_terms = []
    seen = set()
    for t in sorted(terms, key=len, reverse=True):
        if t and t not in seen:
            seen.add(t)
            uniq_terms.append(t)

    # 1) Full-phrase pass (if any phrase present — first item may be phrase)
    phrase_terms = [t for t in uniq_terms if " " in t]
    for p in phrase_terms:
        pat = re.compile(re.escape(p), flags=re.IGNORECASE)
        s = pat.sub(lambda m: f"<mark>{m.group(0)}</mark>", s)

    # 2) Protect existing <mark>...</mark> segments with placeholders to prevent nesting
    placeholders = []
    def _protect(m):
        placeholders.append(m.group(0))
        return f"@@MARK{len(placeholders)-1}@@"
    s = re.sub(r"<mark>.*?</mark>", _protect, s, flags=re.IGNORECASE|re.DOTALL)

    # 3) Highlight single-word tokens with word boundaries, case-insensitive
    single_terms = [t for t in uniq_terms if " " not in t]
    for w in single_terms:
        # \b doesn't work well with hyphenated words; allow hyphen or underscore boundaries too
        pat = re.compile(rf"(?<!\w)({re.escape(w)})(?!\w)", flags=re.IGNORECASE)
        s = pat.sub(r"<mark>\1</mark>", s)

    # 4) Restore placeholders
    def _restore(m):
        idx = int(m.group(1))
        return placeholders[idx]
    s = re.sub(r"@@MARK(\d+)@@", _restore, s)

    return s

def _apply_highlighting(rows: list[dict], terms: list[str]) -> list[dict]:
    """Return a new list with the five target fields highlighted and marked safe for HTML rendering."""
    out = []
    for r in rows:
        rx = dict(r)
        for fld in ["case_law_number","name_of_party","subject_matter1","subject_matter2","summary_head_note"]:
            highlighted = _highlight_text(rx.get(fld) or "", terms)
            # Mark as safe so Jinja doesn't escape <mark> tags
            rx[fld] = Markup(highlighted)
        out.append(rx)
    return out

# ======================= ROUTES =======================
@app.get("/")
def home():
    industries = _distinct("industry_sector")
    sections   = _distinct("section")
    rules      = _distinct("rule")
    parties    = _distinct("name_of_party")
    subjects   = _subjects_grouped()
    is_ai_search = request.args.get('ai_search') == 'true'
    ai_q = (request.args.get("ai_q") or "").strip() if is_ai_search else ""
    sel_industries = request.args.getlist("industry")
    sel_sections   = request.args.getlist("section")
    sel_rules      = request.args.getlist("rule")
    sel_parties    = request.args.getlist("party")
    sel_s1         = request.args.getlist("s1")
    sel_s2         = request.args.getlist("s2")
    q_raw          = (request.args.get("q") or "").strip()
    results, total, pages = [], 0, 1

    # Keep these 3 variables so the template remains unchanged
    ai_statute_text = ""
    ai_circular_text = ""
    ai_error = None

    # Preserve previous per_page behavior: larger page for AI tab
    per_page = 25 if is_ai_search else 10
    COMMON_COLS = _get_common_cols()
    page = max(int(request.args.get("page", 1) or 1), 1)
    offset = (page - 1) * per_page

    def _run_expr(expr):
        with ENGINE.connect() as conn:
            _total = conn.execute(select(func.count()).select_from(Acer).where(expr)).scalar_one()
            _rows = conn.execute(
                select(*COMMON_COLS).where(expr).order_by(Acer.c.id.desc()).limit(per_page).offset(offset)
            ).mappings().all()
        return _total, [dict(r) for r in _rows]

    # ---------- AI tab: DISENGAGED VECTOR —> Phrase-first Plain Word Search with Highlighting ----------
    if is_ai_search and ai_q:
        try:
            ai_explanation_data = get_ai_explanation(ai_q)
            ai_statute_text = ai_explanation_data.get("statute_text", "")
            ai_circular_text = ai_explanation_data.get("circular_text", "")
        except Exception as e:
            ai_error = f"Error preparing AI explanation panel: {e}"
            ai_statute_text = ""
            ai_circular_text = ""

        base_filters = []
        if sel_industries: base_filters.append(Acer.c.industry_sector.in_(sel_industries))
        if sel_sections:   base_filters.append(Acer.c.section.in_(sel_sections))
        if sel_rules:      base_filters.append(Acer.c.rule.in_(sel_rules))
        if sel_parties:    base_filters.append(Acer.c.name_of_party.in_(sel_parties))
        if sel_s1:         base_filters.append(Acer.c.subject_matter1.in_(sel_s1))
        if sel_s2:         base_filters.append(Acer.c.subject_matter2.in_(sel_s2))

        clause_phrase = _phrase_like_clause(ai_q)
        where_phrase  = and_(*(base_filters + [clause_phrase])) if base_filters else clause_phrase
        total, results = _run_expr(where_phrase)

        if total == 0:
            clause_tokens = _word_like_clause(ai_q)
            where_tokens  = and_(*(base_filters + [clause_tokens])) if base_filters else clause_tokens
            total, results = _run_expr(where_tokens)

        pages = (total // per_page) + (1 if total % per_page else 0)

        terms = _word_terms(ai_q)
        results = _apply_highlighting(results, terms)

        q_display = ai_q

    # ---------- Standard search (unchanged) ----------
    elif q_raw or any([sel_industries, sel_sections, sel_rules, sel_parties, sel_s1, sel_s2]):
        per_page = 10
        offset = (page - 1) * per_page

        base_filters = []
        if sel_industries: base_filters.append(Acer.c.industry_sector.in_(sel_industries))
        if sel_sections:   base_filters.append(Acer.c.section.in_(sel_sections))
        if sel_rules:      base_filters.append(Acer.c.rule.in_(sel_rules))
        if sel_parties:    base_filters.append(Acer.c.name_of_party.in_(sel_parties))
        if sel_s1:         base_filters.append(Acer.c.subject_matter1.in_(sel_s1))
        if sel_s2:         base_filters.append(Acer.c.subject_matter2.in_(sel_s2))
        if q_raw:
            where_expr = and_(*(base_filters + [ _nlq_clause(q_raw) ])) if base_filters else _nlq_clause(q_raw)
            total, results = _run_expr(where_expr)
            if total == 0:
                where_expr = and_(*(base_filters + [ like_filters(q_raw) ])) if base_filters else like_filters(q_raw)
                total, results = _run_expr(where_expr)
        else:
            where_expr = and_(*base_filters) if base_filters else text("1=1")
            total, results = _run_expr(where_expr)
        pages = (total // per_page) + (1 if total % per_page else 0)
        q_display = q_raw
    else:
        q_display = ""

    def page_url(n):
        params = {
            'industry': sel_industries, 'section': sel_sections, 'rule': sel_rules,
            'party': sel_parties, 's1': sel_s1, 's2': sel_s2, 'page': n
        }
        if is_ai_search and ai_q:
            params['ai_search'] = 'true'
            params['ai_q'] = ai_q
        else:
            params['q'] = q_raw
        return url_for("home", **params)
    window = 10
    start = max(1, page - (window // 2))
    end = min(pages, start + window - 1)
    start = max(1, end - window + 1)
    page_urls = [{"n": i, "url": page_url(i)} for i in range(start, end + 1)]
    prev_url = page_url(page - 1) if page > 1 else None
    next_url = page_url(page + 1) if page < pages else None

    return render_template(
        "index.html",
        industries=industries, sections=sections, rules=rules, parties=parties, subjects=subjects,
        selected={"industry": set(sel_industries), "section": set(sel_sections),
                  "rule": set(sel_rules), "party": set(sel_parties),
                  "s1": set(sel_s1), "s2": set(sel_s2)},
        q=q_display,
        results=results, page=page, pages=pages, total=total,
        page_urls=page_urls, prev_url=prev_url, next_url=next_url,
        ai_statute_text=ai_statute_text,
        ai_circular_text=ai_circular_text,
        ai_error=ai_error,
        ai_q=ai_q
    )

# --------- Public case detail ---------
@app.get("/case/<int:rid>")
def public_case_detail(rid: int):
    # 1) Fetch the hero case row
    with ENGINE.connect() as conn:
        row = conn.execute(select(Acer).where(Acer.c.id == rid)).mappings().first()
    if not row:
        abort(404)
    r = dict(row)

    # 2) Prepare hero sections (unchanged)
    head_plain      = _strip_all_html(r.get("summary_head_note") or "")
    headnotes_html  = _bold_held_and_prefix(head_plain)
    qa_plain        = _strip_all_html(r.get("question_answered") or "")
    qa_items        = _split_points(qa_plain)
    citations_text  = _strip_all_html(r.get("citation") or "")
    full_case_html  = r.get("full_case_law") or ""

    # 3) Rebuild EXACT SAME result set “slice” the user saw on the list page
    #    (respect filters + q or ai_q, and use identical paging rules)
    args            = request.args
    sel_industries  = args.getlist("industry")
    sel_sections    = args.getlist("section")
    sel_rules       = args.getlist("rule")
    sel_parties     = args.getlist("party")
    sel_s1          = args.getlist("s1")
    sel_s2          = args.getlist("s2")
    q               = (args.get("q") or "").strip()

    is_ai_search    = (args.get("ai_search") == "true")
    ai_q            = (args.get("ai_q") or "").strip() if is_ai_search else ""

    # paging must mirror home(): 25 for AI tab, else 10
    per_page        = 25 if is_ai_search and ai_q else 10
    try:
        page = max(int(args.get("page", 1) or 1), 1)
    except Exception:
        page = 1
    offset          = (page - 1) * per_page

    # base filters (keep as in home())
    clauses = []
    if sel_industries: clauses.append(Acer.c.industry_sector.in_(sel_industries))
    if sel_sections:   clauses.append(Acer.c.section.in_(sel_sections))
    if sel_rules:      clauses.append(Acer.c.rule.in_(sel_rules))
    if sel_parties:    clauses.append(Acer.c.name_of_party.in_(sel_parties))
    if sel_s1:         clauses.append(Acer.c.subject_matter1.in_(sel_s1))
    if sel_s2:         clauses.append(Acer.c.subject_matter2.in_(sel_s2))

    # Query logic:
    # - If AI search: mimic home() AI path = phrase-first, then token fallback
    # - Else if normal q: use _nlq_clause(q) (as home() standard path)
    where_expr = None
    if is_ai_search and ai_q:
        clause_phrase = _phrase_like_clause(ai_q)
        where_phrase  = and_(*(clauses + [clause_phrase])) if clauses else clause_phrase

        # try phrase slice first
        with ENGINE.connect() as conn:
            rows_phrase = conn.execute(
                select(Acer.c.id, Acer.c.case_law_number, Acer.c.name_of_party, Acer.c.date)
                .where(where_phrase).order_by(Acer.c.id.desc())
                .limit(per_page).offset(offset)
            ).mappings().all()
            total_phrase = conn.execute(
                select(func.count()).select_from(Acer).where(where_phrase)
            ).scalar_one()

        if rows_phrase:
            list_results = [dict(x) for x in rows_phrase]
        else:
            # fallback: tokens
            clause_tokens = _word_like_clause(ai_q)
            where_tokens  = and_(*(clauses + [clause_tokens])) if clauses else clause_tokens
            with ENGINE.connect() as conn:
                rows_tokens = conn.execute(
                    select(Acer.c.id, Acer.c.case_law_number, Acer.c.name_of_party, Acer.c.date)
                    .where(where_tokens).order_by(Acer.c.id.desc())
                    .limit(per_page).offset(offset)
                ).mappings().all()
            list_results = [dict(x) for x in rows_tokens]
    else:
        # normal search or pure filters
        if q:
            clauses.append(_nlq_clause(q))
        where_expr = and_(*clauses) if clauses else text("1=1")
        with ENGINE.connect() as conn:
            rows = conn.execute(
                select(Acer.c.id, Acer.c.case_law_number, Acer.c.name_of_party, Acer.c.date)
                .where(where_expr).order_by(Acer.c.id.desc())
                .limit(per_page).offset(offset)
            ).mappings().all()
        list_results = [dict(x) for x in rows]

    # 4) Build the same Search Criteria line, now also showing AI query (if any)
    search_bits = []
    if sel_industries: search_bits.append("Industry: " + ", ".join(sel_industries))
    if sel_sections:   search_bits.append("Section: " + ", ".join(sel_sections))
    if sel_rules:      search_bits.append("Rule: " + ", ".join(sel_rules))
    if sel_parties:    search_bits.append("Party: " + ", ".join(sel_parties))
    if sel_s1:         search_bits.append("Subject 1: " + ", ".join(sel_s1))
    if sel_s2:         search_bits.append("Subject 2: " + ", ".join(sel_s2))
    if is_ai_search and ai_q:
        search_bits.append(f'AI Search: "{ai_q}"')
    elif q:
        search_bits.append(f'Search: "{q}"')

    return render_template(
        "public_case_detail.html",
        r=r,
        headnotes_html=headnotes_html,
        full_case_html=full_case_html,
        citations_text=citations_text,
        qa_items=qa_items,
        search_summary=" | ".join(search_bits),
        list_results=list_results,
    )


# ======================= Summarize Results =======================
def _fetch_rows_for_current_filters(args, cap=60):
    is_ai_search = args.get('ai_search') == 'true'
    ai_q = (args.get('ai_q') or "").strip() if is_ai_search else ""
    sel_industries = args.getlist("industry")
    sel_sections   = args.getlist("section")
    sel_rules      = args.getlist("rule")
    sel_parties    = args.getlist("party")
    sel_s1         = args.getlist("s1")
    sel_s2         = args.getlist("s2")
    q              = (args.get("q") or "").strip()
    clauses = []
    if sel_industries: clauses.append(Acer.c.industry_sector.in_(sel_industries))
    if sel_sections:   clauses.append(Acer.c.section.in_(sel_sections))
    if sel_rules:      clauses.append(Acer.c.rule.in_(sel_rules))
    if sel_parties:    clauses.append(Acer.c.name_of_party.in_(sel_parties))
    if sel_s1:         clauses.append(Acer.c.subject_matter1.in_(sel_s1))
    if sel_s2:         clauses.append(Acer.c.subject_matter2.in_(sel_s2))
    if is_ai_search and ai_q:
        clauses.append(_nlq_clause(ai_q))
    elif q:
        clauses.append(_nlq_clause(q))
    where_expr = and_(*clauses) if clauses else text("1=1")
    with ENGINE.connect() as conn:
        cols = [
            Acer.c.id, Acer.c.case_law_number, Acer.c.name_of_party, Acer.c.date,
            Acer.c.summary_head_note, Acer.c.section, Acer.c.rule,
            Acer.c.type_of_court, Acer.c.state, Acer.c.citation, Acer.c.year
        ]
        rows = conn.execute(
            select(*cols).where(where_expr).order_by(Acer.c.id.desc()).limit(cap)
        ).mappings().all()
    return [dict(r) for r in rows]

def _build_llm_docs(rows):
    docs = []
    for r in rows:
        case_no = str(r.get("case_law_number") or "").strip()
        party   = str(r.get("name_of_party") or "").strip()
        year    = str(r.get("year") or "").strip()
        court   = str(r.get("type_of_court") or "").strip()
        state   = str(r.get("state") or "").strip()
        head    = _strip_all_html(r.get("summary_head_note") or "")
        head    = re.sub(r"\s+", " ", head).strip()
        docs.append({
            "id": r.get("id"),
            "case_no": case_no,
            "party": party,
            "year": year,
            "court": court,
            "state": state,
            "headnote_clean": head[:1200],
        })
    return docs

@app.get("/summarize_results")
def summarize_results():
    try:
        rows = _fetch_rows_for_current_filters(request.args, cap=20)
        if not rows:
            return jsonify({"ok": False, "message": "No case law results to summarize for the current filters."})
        topic = _selected_topic(request.args)
        docs  = _build_llm_docs(rows)
        snippets = "\n".join(
            f"- Case: {d['case_no']} – {d['party']} ({d.get('year','')}) | Court: {d.get('court','')} | State: {d.get('state','')}\n"
            f"  Headnote: {d['headnote_clean']}"
            for d in docs
        )
        user_prompt = f"""
Topic: {topic}
Instruction: Write 2–4 tight paragraphs strictly about the Topic. Base only on these headnotes; ignore unrelated material.
Cite 3–6 cases inline as [{{case_no}} – {{party}} ({{year}})]. No bullet points; paragraphs only. No generic wrap-ups.
Snippets:
{snippets}
Return JSON with keys:
- answer: string (plain text paragraphs, no bullets)
- citations: array of case ids used (from the input 'id')
"""
        data = _openai_chat_json(
            [
                {"role": "system", "content":
                 "You are a concise Indian GST legal analyst. Only use the supplied snippets; if insufficient, say so briefly."},
                {"role": "user", "content": user_prompt}
            ],
            model="gpt-4o-mini", max_tokens=900, temperature=0.3
        )
        answer    = _clean_summary((data or {}).get("answer", ""))
        citations = (data or {}).get("citations", [])
        if not answer.strip():
            return jsonify({"ok": False, "message": "The model returned an empty summary. Try refining filters and retry."})
        return jsonify({"ok": True, "summary": answer, "citations": citations})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500

# ======================= Admin: list/edit/new/export/delete =======================
from sqlalchemy import func, or_

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = (request.form.get('email') or request.form.get('username') or '').strip().lower()
        password = request.form.get('password') or ''
        try:
            computed_hash = hash_password(password)
        except NameError:
            import hashlib
            computed_hash = hashlib.sha256(password.encode()).hexdigest()
        stored_hash = ADMIN_USERS.get(email)
        if stored_hash and stored_hash == computed_hash:
            import time as _t
            session['admin_email'] = email
            session['last_seen_at'] = int(_t.time())
            nxt = request.args.get('next')
            return redirect(nxt or url_for('admin_caselaws'))
        else:
            error = "Invalid email or password."
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('admin_email', None)
    session.pop('last_seen_at', None)
    return redirect(url_for('login'))

@app.get("/admin")
def admin_root():
    return redirect(url_for("admin_caselaws"))

from flask import request, render_template
from sqlalchemy import select, func, and_, or_, text

@app.get("/admin/caselaws")
def admin_caselaws():
    q = (request.args.get("q") or "").strip()
    page = max(int(request.args.get("page", 1)), 1)
    per_page = min(max(int(request.args.get("per_page", 20)), 1), 200)
    offset = (page - 1) * per_page
    mode = request.args.get("mode", "edit")
    edit_item = None
    if 'rid' in request.args or (request.referrer and 'rid=' in request.referrer):
        pass

    where_clause = _nlq_clause(q) if q else text("1=1")

    with ENGINE.connect() as conn:
        total = conn.execute(select(func.count()).select_from(Acer).where(where_clause)).scalar_one()
        rows = conn.execute(
            select(Acer.c.id, Acer.c.case_law_number, Acer.c.name_of_party, Acer.c.date, Acer.c.state)
            .where(where_clause).order_by(Acer.c.id.desc()).limit(per_page).offset(offset)
        ).mappings().all()

        total_case_laws = conn.execute(select(func.count()).select_from(Acer)).scalar_one()

        aar_case_laws = conn.execute(
            select(func.count())
            .select_from(Acer)
            .where(
                and_(
                    func.lower(Acer.c.type_of_court).contains('aar'),
                    ~func.lower(Acer.c.type_of_court).contains('aaar')
                )
            )
        ).scalar_one()

        aaar_case_laws = conn.execute(
            select(func.count())
            .select_from(Acer)
            .where(func.lower(Acer.c.type_of_court).contains('aaar'))
        ).scalar_one()

        # FIX: High Court = case_law_number has 'HC' OR type_of_court has 'High Court' (space) or 'High-Court' (hyphen)
        high_court_case_laws = conn.execute(
            select(func.count())
            .select_from(Acer)
            .where(or_(
                func.lower(Acer.c.case_law_number).contains('hc'),
                func.lower(Acer.c.type_of_court).contains('high court'),
                func.lower(Acer.c.type_of_court).contains('high-court'),
            ))
        ).scalar_one()

        supreme_court_case_laws = conn.execute(
            select(func.count())
            .select_from(Acer)
            .where(or_(
                func.lower(Acer.c.type_of_court).contains('supreme court'),
                func.lower(Acer.c.type_of_court).contains('supreme-court')
            ))
        ).scalar_one()

        rid = request.args.get('rid')
        if rid:
            try:
                edit_item_result = conn.execute(
                    select(Acer).where(Acer.c.id == int(rid))
                ).mappings().first()
                if edit_item_result:
                    edit_item = edit_item_result
            except (ValueError, TypeError):
                edit_item = None
        else:
            edit_item = None

    return render_template(
        "admin_caselaws.html",
        rows=rows,
        page=page,
        per_page=per_page,
        total=total,
        q=q,
        table_name=TABLE_NAME,
        mode=mode,
        edit_item=edit_item,
        total_case_laws=total_case_laws,
        aar_case_laws=aar_case_laws,
        aaar_case_laws=aaar_case_laws,
        high_court_case_laws=high_court_case_laws,
        supreme_court_case_laws=supreme_court_case_laws
    )

@app.route("/admin/caselaws/<int:rid>/edit", methods=["GET", "POST"])
def admin_edit_case(rid: int):
    with ENGINE.connect() as conn:
        row = conn.execute(select(Acer).where(Acer.c.id == rid)).mappings().first()
    if not row:
        abort(404)
    if request.method == "POST":
        data = {}
        for col in ALL_COLS:
            if col == "id": continue
            if col in request.form:
                data[col] = request.form.get(col)
        with ENGINE.begin() as conn:
            conn.execute(update(Acer).where(Acer.c.id == rid).values(**data))
        flash("Saved changes.", "ok")
        return redirect(url_for("admin_caselaws"))
    return render_template("admin_edit.html", r=dict(row), mode="edit")

@app.route("/admin/caselaws/new", methods=["GET", "POST"])
def admin_new_case():
    if request.method == "POST":
        data = {}
        errors = []
        for col in ALL_COLS:
            if col == "id": continue
            if col in request.form:
                data[col] = request.form.get(col)
        with ENGINE.begin() as conn:
            conn.execute(insert(Acer).values(**data))
        flash("New case added.", "ok")
        return redirect(url_for("admin_caselaws"))
    empty = {c: "" for c in ALL_COLS}
    empty["id"] = ""
    return render_template("admin_edit.html", r=empty, mode="new")

def clean_html(raw_html: str) -> str:
    if not isinstance(raw_html, str):
        return str(raw_html) if raw_html is not None else ""
    clean_text = re.sub(r'<[^>]+>', '', raw_html)
    clean_text = clean_text.replace('&nbsp;', ' ')
    clean_text = clean_text.replace('&amp;', '&')
    clean_text = clean_text.replace('<', '<')
    clean_text = clean_text.replace('>', '>')
    clean_text = clean_text.replace('&quot;', '"')
    clean_text = clean_text.replace('&#39;', "'")
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

EXPORT_COLUMN_ORDER = [
    'case_law_number',
    'name_of_party',
    'date',
    'state',
    'year',
    'type_of_court',
    'industry_sector',
    'subject_matter1',
    'subject_matter2',
    'section',
    'rule',
    'case_name',
    'citation',
    'basic_detail',
    'summary_head_note',
    'question_answered'
]

from flask import request, send_file, url_for
from sqlalchemy import select, func, and_, or_, text
import pandas as pd
import io

@app.get("/admin/export_csv")
def admin_export_csv():
    try:
        if 'ALL_COLS' not in globals() or not ALL_COLS:
            print("Error: ALL_COLS is not defined or is empty.")
            return "Error: Column list (ALL_COLS) is missing or empty.", 500
        if 'Acer' not in globals() or Acer is None:
            print("Error: Table object 'Acer' is not defined.")
            return "Error: Table object 'Acer' is missing.", 500
        if 'ENGINE' not in globals() or ENGINE is None:
            print("Error: Database engine 'ENGINE' is not defined.")
            return "Error: Database engine is missing.", 500

        # Always include 'id' for link construction
        base_select_cols = [Acer.c.id]
        export_cols = [c for c in EXPORT_COLUMN_ORDER if hasattr(Acer.c, c)]
        cols_to_select = base_select_cols + [getattr(Acer.c, c) for c in export_cols if c != 'id']

        if not export_cols:
            print("Warning: No valid columns found based on EXPORT_COLUMN_ORDER.")
            df = pd.DataFrame(columns=EXPORT_COLUMN_ORDER)
        else:
            with ENGINE.connect() as conn:
                result = conn.execute(select(*cols_to_select).order_by(Acer.c.id.desc()))
                rows = result.mappings().all()

            df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['id'] + export_cols)
            print(f"Fetched {len(rows)} rows. Cleaning HTML for export...")

            # Clean HTML for specified columns if present
            html_columns_to_clean = [
                'summary_head_note',
                'question_answered',
                'basic_detail',
                'citation'
            ]
            existing_html_columns = [col for col in html_columns_to_clean if col in df.columns]
            if existing_html_columns:
                for col_name in existing_html_columns:
                    df[col_name] = df[col_name].fillna("").apply(lambda x: clean_html(x) if pd.notna(x) else "")

            # Build public case detail link using your route: public_case_detail
            def build_public_link(record_id):
                try:
                    return url_for('public_case_detail', rid=int(record_id), _external=True)
                except Exception:
                    # Fallback path if url_for fails or route name differs
                    base = request.url_root.rstrip('/')
                    return f"{base}/case/{int(record_id)}"

            if 'id' in df.columns:
                df['public_case_link'] = df['id'].apply(build_public_link)
            else:
                df['public_case_link'] = ""

            # Place columns in requested order and append the new link column
            final_column_order = [col for col in EXPORT_COLUMN_ORDER if col in df.columns]
            final_column_order_with_link = final_column_order + (['public_case_link'] if 'public_case_link' in df.columns else [])

            if final_column_order_with_link:
                df = df[final_column_order_with_link]

        buf = io.BytesIO()
        df.to_csv(buf, index=False, encoding="utf-8-sig")
        buf.seek(0)
        return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="cases_cleaned_export.csv")

    except Exception as e:
        print(f"An error occurred during CSV export: {e}")
        import traceback
        traceback.print_exc()
        return f"Internal Server Error during export: {str(e)}", 500

@app.get("/admin/bulk_download")
@app.get("/admin/download")
def admin_bulk_download():
    return admin_export_csv()

@app.route("/admin/delete/<int:rid>", methods=["POST"])
def admin_delete_case(rid):
    with ENGINE.begin() as conn:
        conn.execute(delete(Acer).where(Acer.c.id == rid))
    return redirect(url_for("admin_caselaws", mode="delete"))

# ======================= Admin: BULK UPLOAD (Excel/CSV) =======================
_TEMPLATE_COLUMNS = [
    "case_law_number","name_of_party","state","year","type_of_court",
    "industry_sector","subject_matter1","subject_matter2","section","rule",
    "case_name","date","basic_detail","summary_head_note","citation",
    "question_answered","full_case_law"
]
_SYNONYMS = {
    "case lawnumber":"case_law_number","case law number":"case_law_number","caselaw number":"case_law_number",
    "party":"name_of_party","name of party":"name_of_party",
    "type of court":"type_of_court",
    "industry":"industry_sector","industry sector":"industry_sector",
    "subject matter1":"subject_matter1","subject-matter 1":"subject_matter1","subjectmatter1":"subject_matter1",
    "subject matter2":"subject_matter2","subject-matter 2":"subject_matter2","subjectmatter2":"subject_matter2",
    "basic detail":"basic_detail",
    "summary head note":"summary_head_note","headnote":"summary_head_note","summary":"summary_head_note",
    "question answered":"question_answered","questions answered":"question_answered",
    "full case law":"full_case_law","full text":"full_case_law"
}

def _simp(s:str)->str:
    return re.sub(r"[^a-z0-9]+"," ", (s or "").strip().lower()).strip()

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        sc = _simp(c)
        if sc in _SYNONYMS: rename[c] = _SYNONYMS[sc]
        elif sc in _TEMPLATE_COLUMNS: rename[c] = sc
        else:
            if c in _TEMPLATE_COLUMNS: rename[c] = c
    df = df.rename(columns=rename)
    keep = [c for c in _TEMPLATE_COLUMNS if c in df.columns]
    return df[keep].copy()

def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        original_date_dtype = df["date"].dtype
        print(f"[DEBUG] Original 'date' column dtype: {original_date_dtype}")
        print(f"[DEBUG] Sample 'date' values before coercion:\n{df['date'].head(10)}")
        excel_date_format = "%d-%m-%Y"
        if excel_date_format:
            try:
                df["date"] = pd.to_datetime(df["date"], format=excel_date_format, errors="coerce")
                print(f"[INFO] Parsed dates using explicit format '{excel_date_format}'.")
            except (ValueError, TypeError) as e_format:
                print(f"[WARNING] Failed to parse all dates with format '{excel_date_format}': {e_format}. Falling back to default parser.")
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            print("[INFO] No specific date format provided, using default pd.to_datetime parser.")
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        print(f"[DEBUG] 'date' column dtype after pd.to_datetime: {df['date'].dtype}")
        print(f"[DEBUG] Sample 'date' values after pd.to_datetime:\n{df['date'].head(10)}")
        nat_count_after_to_datetime = df["date"].isna().sum()
        print(f"[DEBUG] Number of NaT/NaN in 'date' after pd.to_datetime: {nat_count_after_to_datetime}")
        nat_count_before_apply = df["date"].isna().sum()
        df["date"] = df["date"].apply(lambda x: None if pd.isna(x) else x)
        nat_count_after_apply = df["date"].isna().sum()
        print(f"[INFO] Date column NaT conversion: {nat_count_before_apply} NaT/NaN values found, {nat_count_before_apply - nat_count_after_apply} converted to None.")
        print(f"[DEBUG] 'date' column dtype after NaT->None conversion: {df['date'].dtype}")
        print(f"[DEBUG] Sample 'date' values after NaT->None conversion:\n{df['date'].head(10)}")

    if "year" in df.columns:
        def _to_year(v):
            if pd.isna(v):
                return None
            try:
                s = str(v).strip()
                if '.' in s and s.replace('.', '', 1).isdigit():
                     float_val = float(s)
                     if float_val.is_integer():
                         s = str(int(float_val))
                if len(s) >= 4 and s[:4].isdigit():
                    return int(s[:4])
                return int(float(s))
            except (ValueError, TypeError):
                return None
        df["year"] = df["year"].map(_to_year)
        if "date" in df.columns:
             mask_years_missing_dates_valid = df["year"].isna() & df["date"].notna()
             if mask_years_missing_dates_valid.any():
                 print(f"[INFO] Deriving {mask_years_missing_dates_valid.sum()} year(s) from parsed 'date' column.")
                 df.loc[mask_years_missing_dates_valid, "year"] = df.loc[mask_years_missing_dates_valid, "date"].dt.year

    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda x: str(x).strip() if not pd.isna(x) else None)
            if c == 'date' and df[c].isin(['NaT']).any():
                 print(f"[ERROR] Found string 'NaT' in 'date' column after object processing!")
                 df[c] = df[c].replace('NaT', None)
    return df

@app.get("/admin/bulk_upload/template")
def admin_bulk_template():
    df = pd.DataFrame(columns=_TEMPLATE_COLUMNS)
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    buf.seek(0)
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="bulk_upload_template.csv")

@app.get("/admin/bulk_template")
def admin_bulk_template_alias():
    return admin_bulk_template()

@app.route("/admin/bulk_upload", methods=["GET", "POST"])
def admin_bulk_upload():
    if request.method == "GET":
        return redirect(url_for("admin_caselaws"))

    if 'file' not in request.files:
        flash("No file part in request.", "err")
        return redirect(url_for("admin_caselaws"))

    f = request.files['file']
    if not f or f.filename == '':
        flash("No file selected.", "err")
        return redirect(url_for("admin_caselaws"))

    filename = secure_filename(f.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in {".xlsx", ".xls", ".csv"}:
        flash("Unsupported file type. Please upload .xlsx, .xls, or .csv", "err")
        return redirect(url_for("admin_caselaws"))

    try:
        data = f.read()
        if ext in {".xlsx", ".xls"}:
            df = pd.read_excel(io.BytesIO(data))
        else:
            df = pd.read_csv(io.BytesIO(data))
    except Exception as e:
        flash(f"Failed to read file: {e}", "err")
        return redirect(url_for("admin_caselaws"))

    if df.empty:
        flash("Uploaded file is empty.", "err")
        return redirect(url_for("admin_caselaws"))

    df = df.replace({np.nan: None})

    def clean_key(s):
        return re.sub(r"\s+", " ", s.strip().lower())

    col_mapping = {}
    synonym_keys = {clean_key(k): v for k, v in _SYNONYMS.items()}

    for col in df.columns:
        c = clean_key(col)
        if c in synonym_keys:
            col_mapping[col] = synonym_keys[c]
        elif c.replace(" ", "_") in _TEMPLATE_COLUMNS:
            mapped = c.replace(" ", "_")
            if mapped in _TEMPLATE_COLUMNS:
                col_mapping[col] = mapped
        elif c in [x.lower() for x in _TEMPLATE_COLUMNS]:
            orig = [x for x in _TEMPLATE_COLUMNS if x.lower() == c][0]
            col_mapping[col] = orig

    if not col_mapping:
        flash("No valid columns found. Please use correct column names.", "err")
        return redirect(url_for("admin_caselaws"))

    df = df.rename(columns=col_mapping)
    df = df[[c for c in _TEMPLATE_COLUMNS if c in df.columns]]
    df = _coerce_types(df)
    df = df.replace({np.nan: None})
    records = df.to_dict("records")

    if not records:
        flash("No data to insert.", "err")
        return redirect(url_for("admin_caselaws"))

    try:
        with ENGINE.begin() as conn:
            case_numbers = [r["case_law_number"] for r in records if r["case_law_number"] is not None]
            if case_numbers:
                conn.execute(delete(Acer).where(Acer.c.case_law_number.in_(case_numbers)))
            conn.execute(insert(Acer), records)
    except Exception as e:
        print(f"Database insert failed: {e}")
        flash(f"Database insert failed: {e}", "err")
        return redirect(url_for("admin_caselaws"))

    flash(f"Successfully uploaded {len(records)} row(s). Duplicates were overwritten.", "ok")
    return redirect(url_for("admin_caselaws"))

# ======================= Contact =======================
@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name    = (request.form.get("name") or "").strip()
        email   = (request.form.get("email") or "").strip()
        subject = (request.form.get("subject") or "").strip() or "Website contact"
        message = (request.form.get("message") or "").strip()

        if not (name and email and message):
            flash("Please fill in your name, email, and message.", "error")
            return redirect(url_for("contact"))

        # Build the email body
        body = (
            "You have a new contact form submission:\n\n"
            f"Name: {name}\n"
            f"Email: {email}\n"
            f"Subject: {subject}\n\n"
            "Message:\n"
            f"{message}\n"
        )

        # Compose MIME email
        msg = MIMEText(body, _subtype="plain", _charset="utf-8")
        msg["Subject"] = Header(f"Contact Form: {subject}", "utf-8")
        msg["From"]    = os.getenv("MAIL_USER", "info@acertax.com")
        msg["To"]      = "info@acertax.com"
        # Let replies go to the sender
        if email:
            msg["Reply-To"] = email

        smtp_user = os.getenv("MAIL_USER", "info@acertax.com")
        smtp_pass = os.getenv("MAIL_PASS")  # ← your Gmail App Password

        try:
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, ["info@acertax.com"], msg.as_string())
            flash("Your message has been sent successfully!", "success")
        except Exception as e:
            # Common cause: no app password / wrong credentials / firewall
            flash(f"Failed to send message: {e}", "error")

        return redirect(url_for("contact"))

    # GET: render page
    team = [
        {"name": "Jayaram Hiregange", "role": "Partner - IDT", "photo": "images/team1.jpg"},
        {"name": "Deepak Rao",        "role": "Partner - IDT", "photo": "images/team2.jpg"},
        {"name": "Bishnu Agarwal",    "role": "Partner - IDT", "photo": "images/team3.jpg"},
        {"name": "Ramya KS",          "role": "Principal Consultant - IDT", "photo": "images/team4.jpg"},
        {"name": "Sandeep Hande",     "role": "Principal Consultant - IDT", "photo": "images/team5.jpg"},
    ]
    if request.method == "POST":
        flash("Thank you! Your message has been received.", "ok")
        return redirect(url_for("contact"))
    return render_template("contact.html", team=team)

# ======================= Meta & Subjects helpers =======================
def _distinct(colname):
    ck = f"distinct:{colname}"
    hit = _meta_cache_get(ck)
    if hit is not None:
        return hit
    with ENGINE.connect() as conn:
        rows = conn.execute(
            select(getattr(Acer.c, colname)).where(
                getattr(Acer.c, colname).isnot(None),
                getattr(Acer.c, colname) != ""
            ).distinct().order_by(getattr(Acer.c, colname).asc())
        ).scalars().all()
    vals = [r for r in rows if r not in (None, "")]
    _meta_cache_set(ck, vals, ttl_sec=300)
    return vals

def _subjects_grouped():
    ck = "subjects"
    hit = _meta_cache_get(ck)
    if hit is not None:
        return hit
    with ENGINE.connect() as conn:
        rows = conn.execute(
            select(Acer.c.subject_matter1, Acer.c.subject_matter2)
            .where(Acer.c.subject_matter1.isnot(None), Acer.c.subject_matter1 != "")
        ).all()
    d = defaultdict(set)
    for s1, s2 in rows:
        k = (s1 or "").strip()
        v = (s2 or "").strip()
        if k:
            if v: d[k].add(v)
            else: d[k] = d[k]
    out = {k: sorted(v) for k, v in sorted(d.items(), key=lambda kv: kv[0].lower())}
    _meta_cache_set(ck, out, ttl_sec=300)
    return out

import logging

@app.route('/_diag/test_login', methods=['POST'])
def _diag_test_login():
    email = (request.form.get('email') or request.form.get('username') or '').strip().lower()
    password = request.form.get('password') or ''
    computed = hash_password(password)
    stored = ADMIN_USERS.get(email)
    logging.warning("LOGIN DIAG email=%r stored=%r computed=%r match=%r",
                    email, stored, computed, stored == computed)
    return {
        "email": email,
        "stored_hash_present": bool(stored),
        "hash_of_entered_password": computed,
        "matches_stored_hash": bool(stored and stored == computed)
    }, 200

# ---------- AI Summary API ----------
# Drop this into app.py (imports at top: from flask import request, jsonify)
# Assumes you already have `_openai_chat_json(messages)` helper.
# No other app behavior is changed.

import re
import html

import os, re, html
from flask import request, jsonify

@app.route('/api/generate_ai_summary', methods=['POST'])
def generate_ai_summary():
    """
    Robust, SDK-version-agnostic summary endpoint.
    - Works with openai>=1.0 (new client) and 0.27/0.28 (legacy).
    - Extracts assistant text directly; doesn't depend on _extract_ai_text.
    """
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON payload"}), 400

    case_text   = (data.get('case_text') or '').strip()
    case_html   = (data.get('case_html') or '').strip()
    case_id     = (str(data.get('case_id') or '').strip() or None)
    case_number = (data.get('case_number') or '').strip()
    case_name   = (data.get('case_name') or '').strip()

    if not case_text and case_html:
        # minimal HTML → text
        s = re.sub(r'(?i)<\s*br\s*/?\s*>', '\n', case_html)
        s = re.sub(r'(?i)</\s*p\s*>', '\n', s)
        s = re.sub(r'(?i)<\s*p[^>]*>', '', s)
        s = re.sub(r'<[^>]+>', '', s)
        s = html.unescape(s)
        s = re.sub(r'\n{3,}', '\n\n', s).strip()
        case_text = s

    # If you want DB fallback, plug it here (kept no-op to avoid wider changes)
    # if not case_text and (case_id or case_number):
    #     case_text = ( _fetch_case_fulltext_from_db(case_id=case_id, case_number=case_number) or "" ).strip()

    if not case_text:
        return jsonify({"error": "No case text provided"}), 400

    # truncate to keep tokens sane
    if len(case_text) > 60000:
        cut = case_text.rfind("\n\n", 0, 60000)
        case_text = (case_text[:cut] if cut != -1 else case_text[:60000]).strip()

    user_prompt = f"""
You are given the full text of an Indian court judgment. Write a clean HTML summary with exactly these section headings in order:
1) Facts
2) Issues
3) Discussion (Arguments & Court's Observations)
4) Decision

Rules:
- Use <h4> for each heading exactly as named above.
- Under each heading, write 1–3 short paragraphs in neutral, precise legal prose. No bullet points.
- Base everything strictly on the provided text; do not add external facts or citations.
- Do not include any preface or conclusion outside these sections.

Case Name: {case_name or '—'}
Case Number: {case_number or '—'}

Full Case Law Text:
{case_text}
""".strip()

    # Ensure API key exists (gives a clear error instead of silent empty content)
    if not os.getenv("OPENAI_API_KEY"):
        return jsonify({"error": "Missing OPENAI_API_KEY in environment"}), 500

    # ---- Call OpenAI (new SDK first, then legacy fallback) ----
    assistant_text = ""
    err_new = None

    # Try new SDK
    try:
        from openai import OpenAI  # openai>=1.0
        client = OpenAI()
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a concise Indian GST legal analyst who writes clean HTML."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        # new SDK object -> extract content
        try:
            assistant_text = (resp.choices[0].message.content or "").strip()
        except Exception:
            # last resort: model_dump
            try:
                d = resp.model_dump()
                assistant_text = (d.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
            except Exception:
                assistant_text = ""
    except Exception as e:
        err_new = e

    # Legacy fallback
    if not assistant_text:
        try:
            import openai  # legacy 0.27/0.28
            model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a concise Indian GST legal analyst who writes clean HTML."},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=1200,
            )
            # legacy object can be OpenAIObject
            try:
                assistant_text = (resp["choices"][0]["message"]["content"] or "").strip()
            except Exception:
                try:
                    assistant_text = (resp.choices[0].message.content or "").strip()
                except Exception:
                    # last resort: dict conversion if available
                    try:
                        d = resp.to_dict_recursive()
                        assistant_text = (d.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
                    except Exception:
                        assistant_text = ""
        except Exception as e_legacy:
            # If both fail, bubble the clearer new-SDK error if we have it, else legacy one
            return jsonify({"error": f"Failed to generate summary: {err_new or e_legacy}"}), 500

    if not assistant_text:
        # Surface a helpful hint instead of a blank error
        hint = "Model returned empty content. Check model name, token limits, or org policy."
        return jsonify({"error": f"AI did not return any content. {hint}"}), 502

    # Tidy minimal: normalize headings & remove lists if any
    def tidy(h: str) -> str:
        if not h: return ""
        # normalize possible bold headings to <h4>
        heads = [
            "Facts",
            "Issues",
            "Discussion (Arguments & Court's Observations)",
            "Decision",
        ]
        for head in heads:
            patterns = [
                rf'(?is)<\s*b[^>]*>\s*{re.escape(head)}\s*<\/\s*b\s*>',
                rf'(?is)<\s*strong[^>]*>\s*{re.escape(head)}\s*<\/\s*strong\s*>',
                rf'(?im)^\s*{re.escape(head)}\s*:\s*',
            ]
            for p in patterns:
                new_h = re.sub(p, f'<h4>{head}</h4>', h, count=1)
                if new_h != h:
                    h = new_h
                    break
        # convert <li> to paragraphs; drop ul/ol
        h = re.sub(r'(?is)<\s*li[^>]*>\s*(.*?)\s*<\/\s*li\s*>', r'<p>\1</p>', h)
        h = re.sub(r'(?is)<\s*\/?\s*(ul|ol)\s*[^>]*>', '', h)
        # strip script/style
        h = re.sub(r'(?is)<\s*(script|style)[^>]*>.*?<\/\s*\1\s*>', '', h)
        # wrap plain text
        if '<' not in h and '>' not in h:
            parts = [f'<p>{html.escape(p.strip())}</p>' for p in h.split('\n') if p.strip()]
            h = ''.join(parts)
        return h.strip()

    summary_html = tidy(assistant_text)
    if not summary_html:
        return jsonify({"error": "AI did not return any content"}), 502

    return jsonify({"summary_html": summary_html}), 200

# ---------- Helpers (lightweight, no external deps) ----------

def _html_to_text(s: str) -> str:
    """Very simple HTML → text; preserves basic newlines."""
    if not s:
        return ""
    # Replace <br> and <p> with newlines
    s = re.sub(r'(?i)<\s*br\s*/?\s*>', '\n', s)
    s = re.sub(r'(?i)</\s*p\s*>', '\n', s)
    s = re.sub(r'(?i)<\s*p[^>]*>', '', s)
    # Strip all other tags
    s = re.sub(r'<[^>]+>', '', s)
    # Unescape entities & collapse extra blank lines
    s = html.unescape(s)
    s = re.sub(r'\n{3,}', '\n\n', s).strip()
    return s

def _soft_truncate(s: str, max_chars: int = 60000) -> str:
    if not s or len(s) <= max_chars:
        return s or ""
    # Cut at nearest paragraph boundary prior to max_chars (if possible)
    cut = s.rfind("\n\n", 0, max_chars)
    if cut == -1:
        cut = max_chars
    return s[:cut].strip()

# ---- Replace your _extract_ai_text with this ----
from collections.abc import Mapping

def _extract_ai_text(ai_resp) -> str:
    """
    Extract text from OpenAI responses across:
    - Legacy openai==0.27/0.28 (OpenAIObject, ChatCompletion)
    - New openai>=1.0 (chat.completions & responses)
    - Dict-like wrappers and our own wrapper
    """
    if ai_resp is None:
        return ""

    # 1) If already a plain string
    if isinstance(ai_resp, str):
        return ai_resp.strip()

    # 2) Normalize to a dict if possible
    d = None
    try:
        if isinstance(ai_resp, Mapping):  # plain dict or dict-like
            d = ai_resp
        elif hasattr(ai_resp, "to_dict_recursive"):  # legacy OpenAIObject
            d = ai_resp.to_dict_recursive()
        elif hasattr(ai_resp, "to_dict"):
            d = ai_resp.to_dict()
        elif hasattr(ai_resp, "model_dump"):  # pydantic models in new SDK
            d = ai_resp.model_dump()
    except Exception:
        d = None

    # 3) Our wrapper might have already placed the content in convenience keys
    if d:
        for key in ("summary_html", "html", "content", "text", "summary", "result", "output_text"):
            val = d.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        # New Chat Completions: choices[0].message.content
        try:
            msg = d["choices"][0]["message"].get("content")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
        except Exception:
            pass

        # Legacy Completions (text)
        try:
            txt = d["choices"][0].get("text")
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
        except Exception:
            pass

        # Responses API style: content as list of parts with "text"
        # e.g., d["choices"][0]["message"]["content"] might be a list of {"type":"text","text": "..."}
        try:
            content = d.get("choices", [{}])[0].get("message", {}).get("content")
            if isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, Mapping):
                        t = c.get("text") or c.get("value") or c.get("content")
                        if isinstance(t, str) and t.strip():
                            parts.append(t.strip())
                if parts:
                    return "\n".join(parts).strip()
        except Exception:
            pass

        # Another Responses shape: d["content"] is a list of parts
        try:
            cont_list = d.get("content")
            if isinstance(cont_list, list):
                parts = []
                for c in cont_list:
                    if isinstance(c, Mapping):
                        t = c.get("text") or c.get("value") or c.get("content")
                        if isinstance(t, str) and t.strip():
                            parts.append(t.strip())
                if parts:
                    return "\n".join(parts).strip()
        except Exception:
            pass

    # 4) Last resort: access attributes (legacy objects) without dict conversion
    #    ai_resp.choices[0].message.content
    try:
        ch0 = ai_resp.choices[0]
        msg = getattr(getattr(ch0, "message", None), "content", None)
        if isinstance(msg, str) and msg.strip():
            return msg.strip()
        # or ai_resp.choices[0].text
        txt = getattr(ch0, "text", None)
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
    except Exception:
        pass

    return ""


def _tidy_summary_html(h: str) -> str:
    if not h:
        return ""
    # Ensure required headings exist; if the model used bold text, normalize to <h4>
    def normalize_heading(text, canonical):
        # replace occurrences like <b>Facts</b>, <strong>Facts</strong>, or plain "Facts:" at start
        patterns = [
            rf'(?is)<\s*b[^>]*>\s*{re.escape(canonical)}\s*<\/\s*b\s*>',
            rf'(?is)<\s*strong[^>]*>\s*{re.escape(canonical)}\s*<\/\s*strong\s*>',
            rf'(?im)^\s*{re.escape(canonical)}\s*:\s*'
        ]
        for p in patterns:
            h_nonlocal = re.sub(p, f'<h4>{canonical}</h4>', text, count=1)
            if h_nonlocal != text:
                return h_nonlocal
        return text

    for head in ["Facts", "Issues", "Discussion (Arguments & Court's Observations)", "Decision"]:
        h = normalize_heading(h, head)

    # Convert accidental lists to paragraphs (no bullets rule)
    h = re.sub(r'(?is)<\s*li[^>]*>\s*(.*?)\s*<\/\s*li\s*>', r'<p>\1</p>', h)
    h = re.sub(r'(?is)<\s*\/?\s*(ul|ol)\s*[^>]*>', '', h)

    # Remove script/style just in case
    h = re.sub(r'(?is)<\s*(script|style)[^>]*>.*?<\/\s*\1\s*>', '', h)

    # Basic sanity: wrap plain text in paragraphs if no tags at all
    if '<' not in h and '>' not in h:
        parts = [f'<p>{html.escape(p.strip())}</p>' for p in h.split('\n') if p.strip()]
        h = ''.join(parts)

    return h.strip()


# ---------- Optional DB fetch (wire up only if you need it) ----------
def _fetch_case_fulltext_from_db(case_id=None, case_number=None) -> str:
    """
    Replace the internals with your actual ORM/DB lookup.
    The function should return the FULL case text (plain text) for the given ID/number.
    This is intentionally a stub so we don't alter your existing DB code/objects.

    Example (SQLAlchemy pseudo-code):
        from your_models import AcerDetails
        q = AcerDetails.query
        if case_id:
            rec = q.filter_by(id=case_id).first()
        elif case_number:
            rec = q.filter_by(case_law_number=case_number).first()
        else:
            rec = None
        if rec:
            # Prefer a dedicated plain-text field if you have one; else strip tags from HTML
            text = (rec.full_case_text or _html_to_text(rec.full_case_html) or "").strip()
            return text
        return ""
    """
    return ""

if __name__ == "__main__":
    app.run(debug=True)
