from flask import Flask, request, render_template, abort, redirect, url_for, flash, send_file, jsonify, render_template_string
from sqlalchemy import create_engine, MetaData, Table, select, or_, func, text, and_, insert, update, delete
# Import the `cast` and `float` functions for potential vector column casting
from sqlalchemy import cast, Float
from bs4 import BeautifulSoup
from werkzeug.utils import secure_filename
import os, re, io, ssl, json, time, hashlib, math
import pandas as pd
from collections import defaultdict
import numpy as np


# ======================= FLASK & UPLOAD CONFIG =======================
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64MB upload cap
ALLOWED_UPLOADS = {".xlsx", ".xls", ".csv"}

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

# ======================= OPENAI CONFIG =======================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_env_model = (os.getenv("OPENAI_MODEL") or "").strip()
_ALLOWED = {"gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"}
OPENAI_MODEL = _env_model if _env_model in _ALLOWED else "gpt-4o-mini"
# Embeddings
EMBED_MODEL = (os.getenv("OPENAI_EMBED_MODEL") or "text-embedding-3-small").strip()
# Try to auto-detect a column that stores embeddings
EMBED_COL_CANDIDATES = [
    "embedding","embeddings","text_embedding","text_embeddings",
    "summary_embedding","headnote_embedding","openai_embedding",
    "vector","vec","text_vector"
]
# Hybrid ranking weights (tweakable via env)
W_VEC   = float(os.getenv("HYBRID_W_VEC", "0.7"))
W_LEX   = float(os.getenv("HYBRID_W_LEX", "0.3"))
SCORE_MIN = float(os.getenv("HYBRID_SCORE_MIN", "0.18"))  # threshold to drop very weak matches
CANDIDATE_PRIMARY = int(os.getenv("HYBRID_CANDIDATES_PRIMARY", "400"))
CANDIDATE_WIDE    = int(os.getenv("HYBRID_CANDIDATES_WIDE", "800"))
# Vector-only recall sweep controls
VECTOR_SWEEP_LIMIT = int(os.getenv("VECTOR_SWEEP_LIMIT", "1200"))  # how many rows to scan for pure vector recall
VECTOR_TOPK        = int(os.getenv("VECTOR_TOPK", "250"))          # top-k vector-only hits to merge
VEC_ONLY_MIN       = float(os.getenv("VEC_ONLY_MIN", "0.55"))      # min cosine to keep vector-only hits (Lowered as discussed)


# Predefined admin users: email -> hashed_password
# Password: "admin123"
ADMIN_USERS = {
    "admin@acergst.com": "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",  # sha256("admin123")
    "support@acergst.com": "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
}

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Optional: Add more admins programmatically
# ADMIN_USERS["new@admin.com"] = hash_password("yourpassword")

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
    """
    Get a single embedding vector for text. Falls back cleanly if API is missing.
    """
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
    """
    Gate all /admin* routes behind login and enforce a simple idle timeout.
    - If not logged in, redirect to /login?next=...
    - If idle for >30 minutes, force re-login.
    """
    # Local import so this def is self-contained for copy-paste
    import time as _t

    path = (request.path or "")
    if not path.startswith("/admin"):
        return  # only guard the /admin prefix

    # If there is no authenticated admin in session, bounce to login
    if 'admin_email' not in session:
        return redirect(url_for('login', next=request.url))

    # Idle-timeout check (30 minutes). Adjust seconds (1800) if you want.
    now = int(_t.time())
    last_seen = session.get('last_seen_at')
    if last_seen and (now - last_seen) > 1800:
        # Too long idle — clear session and ask to re-login
        session.pop('admin_email', None)
        session.pop('last_seen_at', None)
        return redirect(url_for('login', next=request.url))

    # Refresh the activity timestamp on any /admin request
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
    """
    OR across SEARCHABLE_COLS using LIKE.
    SAFE: never degrade to a match-all for non-empty queries.
    """
    q = (q or "").strip()
    if not q:
        # User typed nothing (or query collapsed) -> do NOT dump the entire table
        return text("0=1")
    terms = []
    for col in SEARCHABLE_COLS:
        if hasattr(Acer.c, col):
            terms.append(getattr(Acer.c, col).like(f"%{q}%"))
    # If no valid searchable columns were found, fail closed.
    return or_(*terms) if terms else text("0=1")

# ---- Minimal stopwords (we rely on ranking rather than heavy filtering)
_STOPWORDS = {"a","an","the","of","and","or","in","on","to","by","for","with","under","about",
              "this","that","these","those","some"}
_KEEP_SHORT = {"itc","gst","rcm","hsn","pos","b2b","b2c"}
# Domain phrases we treat as MUST if present in the query
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
    """
    Phrase-first extraction with minimal stopwording.
    Keeps multi-word domain phrases; remaining single tokens are kept
    unless they are very short or in a tiny stopword set.
    """
    if not q:
        return []
    s = q.lower()
    s = re.sub(r"[^\w\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    kept, seen = [], set()
    # 1) Capture domain phrases first (multi-word keepers)
    s_for_phrase = " " + s + " "
    for ph in sorted(_DOMAIN_PHRASES, key=len, reverse=True):
        ph_norm = " " + ph + " "
        if ph_norm in s_for_phrase:
            if ph not in seen:
                kept.append(ph); seen.add(ph)
            s_for_phrase = s_for_phrase.replace(ph_norm, " ")
    # 2) Residual single tokens (minimal stopwords)
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
        # Light fallback: try simple tokenization; if still empty, fail closed via like_filters
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
    # Defensive: never return a match-all here
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

# ---------- Vector helpers ----------
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

# --- New Helper Function for Direct Vector Search (Clean Implementation) ---
def _vector_search_direct(ai_q: str, base_filters: list, limit: int = 200, offset: int = 0) -> tuple[list[dict], int]:
    """
    Performs a direct vector similarity search.
    1. Fetches candidates based on base_filters and non-null embedding.
    2. Generates embedding for the user's query (ai_q).
    3. Calculates cosine similarity for each candidate.
    4. Applies VEC_ONLY_MIN threshold.
    5. Sorts by similarity score (desc).
    6. Returns a tuple: (paginated_results_list, total_count_of_filtered_results)
    """
    if not ai_q.strip():
        return [], 0
    # 1. Detect the embedding column name in the database table
    embed_col_name = _detect_embedding_column_name()
    if not embed_col_name:
        print("[Vector Direct] Error: No embedding column detected.")
        return [], 0
    embed_col = getattr(Acer.c, embed_col_name)
    # 2. Generate embedding for the user's natural language query
    q_key = "search_embed_direct:" + hashlib.sha256(ai_q.encode("utf-8")).hexdigest()
    q_vec = _ai_cache_get(q_key)
    if q_vec is None:
        q_vec = _openai_embed(ai_q) # Embed the raw user query
        if q_vec is not None:
            _ai_cache_set(q_key, q_vec, ttl_sec=1800) # Cache for 30 mins
            print(f"[Vector Direct] Query embedding generated (len={len(q_vec)})")
        else:
            print("[Vector Direct] Error: Could not generate query embedding.")
            return [], 0
    # 3. Prepare database query to fetch candidates
    #    Base filters from UI (should be empty if none selected)
    #    + Mandatory check for non-null embedding column
    where_conditions = [embed_col.isnot(None)] # Ensure row has an embedding
    if base_filters:
        where_conditions.extend(base_filters)
    where_expr = and_(*where_conditions) if where_conditions else text("1=1")
    # 4. Fetch candidate rows (data + embedding) from the database
    with ENGINE.connect() as conn:
        cols_to_select = _get_common_cols() + [embed_col] # Include embedding column
        # Fetch all matching candidates (filtering/sorting happens in Python)
        # Consider LIMITing if dataset is huge and this becomes a bottleneck
        result_proxy = conn.execute(
            select(*cols_to_select).where(where_expr).order_by(Acer.c.id.desc())
        )
        candidate_rows = result_proxy.mappings().all()
    print(f"[Vector Direct] Fetched {len(candidate_rows)} candidate rows with embeddings.")
    # 5. Score candidates using cosine similarity
    scored_candidates = []
    for row in candidate_rows:
        db_embedding_data = row.get(embed_col_name)
        if db_embedding_data is None:
            continue # Defensive check
        # Parse the stored embedding (JSON string -> list of floats)
        db_vec = _parse_vec(db_embedding_data)
        if db_vec is None:
            # print(f"[Vector Direct] Warning: Could not parse embedding for row ID {row.get('id')}")
            continue # Skip rows with unparsable embeddings
        # Calculate similarity
        similarity_score = _cosine(q_vec, db_vec)
        # Apply minimum threshold early to reduce list size
        if similarity_score >= VEC_ONLY_MIN:
            row_dict = dict(row)
            row_dict["_score"] = float(similarity_score)
            scored_candidates.append(row_dict)
    print(f"[Vector Direct] {len(scored_candidates)} candidates passed similarity threshold ({VEC_ONLY_MIN}).")
    # 6. Sort candidates by similarity score descending
    scored_candidates.sort(key=lambda x: x["_score"], reverse=True)
    # 7. Calculate the total number of results BEFORE pagination
    total_count = len(scored_candidates)
    # 8. Apply pagination (offset and limit) to get the final page of results
    paginated_results = scored_candidates[offset : offset + limit]
    print(f"[Vector Direct] Returning {len(paginated_results)} results (offset={offset}, limit={limit}). Total available: {total_count}")
    # Return the paginated results and the total count for correct pagination links
    return paginated_results, total_count

def _parse_vec(val):
    """
    Accept JSON array, Python list, or comma-separated string; return list[float] or None.
    """
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
        # JSON array?
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [float(x) for x in arr]
            except Exception:
                pass
        # CSV floats?
        try:
            parts = [p for p in re.split(r"[,\s]+", s) if p]
            if len(parts) >= 8:  # guard tiny junk
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
    """
    Weighted lexical score over key fields:
    - case_law_number (2.0)
    - name_of_party    (1.5)
    - subject_matter1/2(1.2 each)
    - summary_head_note(1.0, cleaned)
    Normalized to 0..1
    """
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
    # Tokenize fields
    tokenized = {k: _tokenize(v) for k, v in fields.items()}
    # Score presence
    max_possible = sum(weights.values()) * len(q_tokens)
    if max_possible == 0:
        return 0.0
    score = 0.0
    for tok in q_tokens:
        for fld, toks in tokenized.items():
            if tok in toks:
                score += weights[fld]
    # Small bonus for digit sequences in case number appearing in query
    if fields["case_law_number"]:
        digits_in_case = re.findall(r"\d+", fields["case_law_number"])
        for d in digits_in_case[:3]:
            if d and d in " ".join(q_tokens):
                score += 0.6
                break
    # Clamp
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
        ln = re.sub(r'^\s*[\*\-\u2022\u00a0\u00a2]+\s*', '', ln) # Handle common bullet chars
        ln = re.sub(r'^\s*\d+\.\s*', '', ln)
        lines.append(ln)
    text = "\n".join(lines)
    text = re.sub(r'(?im)^\s*(overall|in summary|in conclusion|to conclude|in short)\b.*$', '', text)
    text = re.sub(r'\n{3,}', '\n', text).strip()
    return text

# ======================= AI EXPLANATION (ENHANCED: Statute + Circulars) =======================
# ======================= AI EXPLANATION (DETAILED + NO CASE NAMES) - ENHANCED =======================
# ======================= AI EXPLANATION (MODIFIED: Specific Headings & Disclaimer) =======================
def get_ai_explanation(prompt_text):
    """
    Generates a two-part AI explanation with specific headings and a disclaimer.
    Part A: Relevant Provisions (Sections/Rules).
    Part B: Applicable Circulars/Notifications.
    Adds a standard disclaimer at the end.
    Caches the combined result.
    """
    args = request.args
    # Create a unique cache key based ONLY on the query text for this specific format
    key_src = "|".join([
        "ai_explain_headings_disclaimer_v1", (prompt_text or "").strip(),
        # Filter context explicitly removed from cache key and prompt as requested previously
    ])
    cache_key = "ai_explain_hd:" + hashlib.sha256(key_src.encode("utf-8")).hexdigest()
    cached = _ai_cache_get(cache_key)
    if cached is not None:
        print(f"[AI Explain HD] Cache HIT for key: {cache_key[:20]}...")
        return cached # Return the cached dict directly

    print(f"[AI Explain HD] Cache MISS for key: {cache_key[:20]}... Generating...")

    user_query = (prompt_text or '').strip()

    # --- Construct the prompt for the AI ---
    # Instruct the AI to generate content under specific headings and add the disclaimer.
    # Inside get_ai_explanation function
    ai_prompt = (
        "You are an expert on Indian GST law. Please structure your response exactly as follows:\n\n"
        "A. Relevant Provisions\n"
        "<Write 2-3 well-structured paragraphs here focusing ONLY on the LATEST and relevant sections and rules "
        "of the CGST/IGST Acts that pertain to the user's query. "
        "Ensure the paragraphs are clear, justified, and directly address the statutory provisions. "
        "Do NOT mention or invent ANY case law, case names, parties, judges, or citations.>\n\n"
        "B. Applicable Circulars/Notifications\n"
        "<Write 2-3 well-structured paragraphs here detailing the LATEST and specific circulars and notifications "
        "issued by the GST Council or CBIC that are applicable to the user's query. "
        "Mention the circular/notification number and date where known. "
        "Ensure the paragraphs are clear, justified, and directly address these administrative guidelines. "
        "Do NOT mention or invent ANY case law, case names, parties, judges, or citations.>\n\n"
        "---\n"
        "It is advisable for the user to refer to the latest provisions of the law.\n" # Keep this reminder
        f"User Query: {user_query}"
    )
    # ... inside the system message ...
    {"role": "system", "content": (
        "You are a precise Indian GST legal analyst. "
        "Prioritize providing information based on the LATEST versions of the CGST/IGST Acts, Rules, Notifications, and Circulars. "
        # ... rest of system message ...
    )}

    statute_text = ""
    circular_text = ""
    disclaimer = "It is advisable for the user to refer to the latest provisions of the law."

    try:
        print(f"[AI Explain HD] Calling OpenAI API...")
        # Single call to generate the full structured response
        full_response = _openai_chat(
            [
                {"role": "system", "content": (
                    "You are a precise Indian GST legal analyst. "
                    "Follow the EXACT output format requested: "
                    "Start with 'A. Relevant Provisions', followed by paragraphs, "
                    "then 'B. Applicable Circulars/Notifications', followed by paragraphs, "
                    "then '---', then the disclaimer. "
                    "Focus strictly on statutory provisions (Acts, Rules) and administrative documents (Circulars, Notifications). "
                    "Never cite or name cases. Ensure paragraphs are well-structured."
                )},
                {"role": "user", "content": ai_prompt},
            ],
            max_tokens=1200, # Adjust if needed
            temperature=0.3,
        )
        full_response = (full_response or "").strip()

        if not full_response:
             raise ValueError("OpenAI returned an empty response.")

        # --- Parse the AI response ---
        # 1. Split by the section headings
        # Find the start of section B
        section_b_start = full_response.find("B. Applicable Circulars/Notifications")
        if section_b_start == -1:
            # If B section not found, assume whole response is A section
            statute_text = full_response
            circular_text = "Applicable circulars and notifications could not be generated."
            print("[AI Explain HD] Warning: Could not parse section B. Assigning all text to section A.")
        else:
            # Extract A section (text before B section)
            statute_text = full_response[:section_b_start].replace("A. Relevant Provisions", "", 1).strip()

            # Find the start of the disclaimer separator
            disclaimer_start = full_response.find("---", section_b_start)
            if disclaimer_start == -1:
                # If separator not found, take everything after B section as B section
                circular_text = full_response[section_b_start + len("B. Applicable Circulars/Notifications"):].strip()
                # Add disclaimer manually if not found
                # Note: The AI prompt asks for it, but we add it here too to be safe.
            else:
                # Extract B section (text between B heading and disclaimer separator)
                b_end_index = disclaimer_start
                b_content = full_response[section_b_start + len("B. Applicable Circulars/Notifications"):b_end_index].strip()
                # Extract potential disclaimer text (text after separator)
                potential_disclaimer = full_response[disclaimer_start + len("---"):].strip()

                circular_text = b_content

                # Optional: Verify the disclaimer matches or log if it's different
                # if potential_disclaimer and potential_disclaimer != disclaimer:
                #     print(f"[AI Explain HD] AI generated disclaimer differs slightly: '{potential_disclaimer}'")


    except Exception as e:
        error_msg = f"Error generating AI explanation: {e}"
        print(error_msg)
        # Return error messages for each part
        statute_text = f"Error generating statutory provisions: {e}"
        circular_text = f"Error generating circulars/notifications: {e}"
        # Disclaimer is static, so it remains the default string

    # --- Cache and Return Result ---
    result = {
        "statute_text": statute_text,
        "circular_text": circular_text,
        "disclaimer": disclaimer, # Include disclaimer in the returned data
        "cached": False
    }
    _ai_cache_set(cache_key, result, ttl_sec=900) # Cache for 15 minutes
    print(f"[AI Explain HD] Generated and cached result for key: {cache_key[:20]}...")
    return result

# ======================= HYBRID AI SEARCH (vector + lexical + vector fallback) =======================
# --- Keep the _get_common_cols and _run_query functions as they are ---
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

# --- Keep the _hybrid_rank function for potential future use or reference, but it's not called anymore in the AI search path ---
def _hybrid_rank(ai_q: str, ai_text: str, base_filters):
    """
    Returns scored rows (list of dicts with '_score' added), already filtered by SCORE_MIN.
    """
    # Build merged text for lexical candidate recall
    merged_search_text = _merge_ai_and_user_keywords(ai_text, ai_q)
    # Detect embedding column
    embed_col_name = _detect_embedding_column_name()
    embed_col = getattr(Acer.c, embed_col_name) if embed_col_name else None
    # Candidate set 1: NLQ on merged tokens (phrase-must + token-should)
    where_expr_1 = and_(*(base_filters + [ _nlq_clause(merged_search_text) ])) if base_filters else _nlq_clause(merged_search_text)
    add_cols = [embed_col] if embed_col is not None else []
    cand = _run_query(where_expr_1, add_cols=add_cols, limit=CANDIDATE_PRIMARY)
    # Candidate set 2 (wider): plain LIKE on raw query if too few
    if len(cand) < 50:
        where_expr_2 = and_(*(base_filters + [ like_filters(ai_q) ])) if base_filters else like_filters(ai_q)
        cand2 = _run_query(where_expr_2, add_cols=add_cols, limit=CANDIDATE_WIDE)
        # merge dedupe
        seen = {c["id"] for c in cand}
        for r in cand2:
            if r["id"] not in seen:
                cand.append(r); seen.add(r["id"])
    # Prepare query embedding using key phrase(s)
    sem_q = _build_semantic_query(ai_q, ai_text)
    q_key = "embed:" + hashlib.sha256(sem_q.encode("utf-8")).hexdigest()
    q_vec = _ai_cache_get(q_key)
    if q_vec is None:
        q_vec = _openai_embed(sem_q)
        if q_vec is not None:
            _ai_cache_set(q_key, q_vec, ttl_sec=1800)
    # Vector-only fallback recall if lexical recall weak
    if (not cand or len(cand) < 10) and embed_col is not None and q_vec is not None:
        # Respect filters; scan a bounded slice for vector similarity
        sweep_where = and_(*base_filters) if base_filters else text("1=1")
        sweep_rows = _run_query(sweep_where, add_cols=[embed_col], limit=VECTOR_SWEEP_LIMIT, offset=0)
        vec_hits = []
        for r in sweep_rows:
            rv = r.get(embed_col_name)
            v = _parse_vec(rv)
            if v:
                sim = _cosine(q_vec, v)
                if sim >= VEC_ONLY_MIN:
                    rx = dict(r); rx["_vec_only_sim"] = float(sim)
                    vec_hits.append(rx)
        vec_hits.sort(key=lambda x: x["_vec_only_sim"], reverse=True)
        vec_hits = vec_hits[:VECTOR_TOPK]
        # merge into cand by id
        seen = {c["id"] for c in cand}
        for r in vec_hits:
            if r["id"] not in seen:
                cand.append(r); seen.add(r["id"])
    # If still nothing found, bail
    if not cand:
        return []
    q_tokens = set(_nlq_keywords(ai_q)) | _tokenize(ai_q)
    scored = []
    for r in cand:
        # Vector similarity
        vec_sim = 0.0
        if q_vec is not None and embed_col_name:
            rv = r.get(embed_col_name)
            v = _parse_vec(rv)
            if v:
                vec_sim = _cosine(q_vec, v)
        # Lexical score (weighted over case no, party, s1, s2, headnote)
        lex = _lexical_match_score(q_tokens, r)
        # Hybrid score
        score = (W_VEC * vec_sim) + (W_LEX * lex)
        # Drop ultra-low score rows unless we have very few candidates
        if score >= SCORE_MIN or len(cand) <= 30:
            rx = dict(r)
            rx["_score"] = round(float(score), 6)
            scored.append(rx)
    # Sort by score desc, then id desc
    scored.sort(key=lambda x: (x["_score"], x.get("id", 0)), reverse=True)
    return scored

def get_ai_explanation(prompt_text):
    """
    Generates a two-part AI explanation: Statutory Provisions and Circulars/Notifications.
    Caches the combined result.
    """
    args = request.args
    # Create a unique cache key based on the query and filters
    # Removed filter context from cache key and prompt as requested
    key_src = "|".join([
        "ai_explain_modified_v1", (prompt_text or "").strip(),
        # Filter context removed from cache key
    ])
    cache_key = "ai_explain_modified:" + hashlib.sha256(key_src.encode("utf-8")).hexdigest()
    cached = _ai_cache_get(cache_key)
    if cached is not None:
        print(f"[AI Explain Modified] Cache HIT for key: {cache_key[:20]}...")
        return cached # Return the cached dict directly

    print(f"[AI Explain Modified] Cache MISS for key: {cache_key[:20]}... Generating...")

    user_query = (prompt_text or '').strip()

    # --- 1. Generate Statutory Provisions (Section A) ---
    statute_prompt = (
        "You are an expert on Indian GST. "
        "Write ONLY the response in the following format:\n\n"
        "A. Relevant Provisions\n"
        "<Write 2-3 paragraphs strictly about the relevant CGST/IGST Acts and Rules related to the user's query. "
        "Focus on sections and rules. Do NOT mention or invent ANY case names, parties, judges, or case citations. "
        "NO bullet points, NO generic wrap-ups; just crisp paragraphs.>\n\n"
        "B. Relevant Circulars/Notifications\n"
        "<Write 2-3 paragraphs strictly about applicable GST Circulars and Notifications related to the user's query. "
        "Mention specific circular/notification numbers and dates if known. "
        "Do NOT mention or invent ANY case names, parties, judges, or case citations. "
        "NO bullet points, NO generic wrap-ups; just crisp paragraphs.>\n\n"
        "User Query: " + user_query
    )

    statute_text = ""
    circular_text = ""
    try:
        # Use a single call to generate both parts as instructed
        full_response = _openai_chat(
            [
                {"role": "system", "content": "You are a precise Indian GST legal analyst. Follow the exact output format requested by the user. Focus on statutory provisions and circulars/notifications. Never cite or name cases."},
                {"role": "user", "content": statute_prompt},
            ],
            max_tokens=1200, temperature=0.3, # Increased tokens for two sections
        )
        full_response = full_response.strip()
        # Simple parsing: Assume sections A and B are clearly marked
        # Find the start of section B
        section_b_index = full_response.find("B. Relevant Circulars/Notifications")
        if section_b_index != -1:
            statute_text = full_response[:section_b_index].replace("A. Relevant Provisions", "").strip()
            circular_text = full_response[section_b_index + len("B. Relevant Circulars/Notifications"):].strip()
        else:
            # Fallback if parsing fails, assume whole response is statute part
            statute_text = full_response
            circular_text = "Explanation of relevant circulars and notifications could not be generated separately."

    except Exception as e:
        error_msg = f"Error generating AI explanation: {e}"
        print(error_msg)
        statute_text = f"Error generating statutory provisions: {e}"
        circular_text = f"Error generating circulars/notifications: {e}"

    # --- Cache and Return Result ---
    result = {
        "statute_text": statute_text,
        "circular_text": circular_text,
        "cached": False # Indicate it was just generated
    }
    _ai_cache_set(cache_key, result, ttl_sec=900) # Cache for 15 minutes
    print(f"[AI Explain Modified] Generated and cached result for key: {cache_key[:20]}...")
    return result

# ... (previous code remains unchanged until the home route handler) ...

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
    # Initialize separate AI response components
    ai_statute_text = ""
    ai_circular_text = ""
    ai_error = None # To capture any AI generation errors for display

    # --- CHANGE 3: Increase per_page for AI search ---
    # Set per_page based on search type
    if is_ai_search:
        per_page = 25 # Increased for AI search
    else:
        per_page = 10 # Standard search
    # --- END CHANGE 3 ---

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

    # ---------- AI search (MODIFIED: Vector Similarity + AI Explanation) ----------
    if is_ai_search and ai_q:
        # --- CHANGE 1: Generate the Two-Part AI Explanation ---
        try:
            ai_explanation_data = get_ai_explanation(ai_q)
            # Unpack the separate parts for the template
            ai_statute_text = ai_explanation_data.get("statute_text", "")
            ai_circular_text = ai_explanation_data.get("circular_text", "")
            print(f"[Modified AI Search] Generated explanation for query: '{ai_q}'")
        except Exception as e:
            error_msg = f"Error generating AI explanation: {e}"
            print(error_msg)
            ai_error = error_msg # Store error to display in template
            ai_statute_text = ""
            ai_circular_text = ""

        # --- Vector Search Logic (Enhanced for Relevance - Addresses CHANGE 2) ---
        print(f"[Vector AI Search] Processing query: '{ai_q}'")
        # --- 1. Generate embedding for the user's query ---
        q_key = "search_embed_ai_route:" + hashlib.sha256(ai_q.encode("utf-8")).hexdigest()
        q_vec = _ai_cache_get(q_key)
        if q_vec is None:
            q_vec = _openai_embed(ai_q) # Embed the raw user query
            if q_vec is not None:
                _ai_cache_set(q_key, q_vec, ttl_sec=1800) # Cache for 30 mins
                print(f"[Vector AI Search] Query embedding generated (len={len(q_vec)})")
            else:
                print("[Vector AI Search] Error: Could not generate query embedding.")
                ai_error = ai_error + " Could not generate query vector." if ai_error else "Could not generate query vector."
                results = []
                total = 0
                pages = 1
                q_display = ai_q
                # Proceed to render with error

        # --- 2. Proceed only if embedding was generated ---
        if q_vec is not None:
            # --- 3. Detect the embedding column name in the database table ---
            embed_col_name = _detect_embedding_column_name() # Uses Acer table definition
            if not embed_col_name:
                print("[Vector AI Search] Error: No embedding column detected in acer_details.")
                ai_error = ai_error + " No embedding column found in database." if ai_error else "No embedding column found in database."
                results = []
                total = 0
                pages = 1
                q_display = ai_q
                # Proceed to render with error
            else:
                embed_col = getattr(Acer.c, embed_col_name)
                print(f"[Vector AI Search] Using embedding column: {embed_col_name}")

                # --- 4. Build base filters from UI selections ---
                # --- CHANGE 2 Enhancement: Stricter filtering ---
                # Ensure candidate rows have embeddings and match UI filters.
                # Consider increasing VEC_ONLY_MIN if random results persist.
                base_filters = [embed_col.isnot(None)] # Mandatory: ensure row has an embedding
                if sel_industries: base_filters.append(Acer.c.industry_sector.in_(sel_industries))
                if sel_sections:   base_filters.append(Acer.c.section.in_(sel_sections))
                if sel_rules:      base_filters.append(Acer.c.rule.in_(sel_rules))
                if sel_parties:    base_filters.append(Acer.c.name_of_party.in_(sel_parties))
                if sel_s1:         base_filters.append(Acer.c.subject_matter1.in_(sel_s1))
                if sel_s2:         base_filters.append(Acer.c.subject_matter2.in_(sel_s2))

                where_expr = and_(*base_filters) if base_filters else embed_col.isnot(None)
                print(f"[Vector AI Search] Base filters applied (including non-null embedding check).")

                # --- 5. Fetch candidate rows (data + embedding) from the database ---
                with ENGINE.connect() as conn:
                    cols_to_select = _get_common_cols() + [embed_col] # Include embedding column
                    # Fetch candidates based on filters. Sorting happens in Python.
                    result_proxy = conn.execute(
                        select(*cols_to_select).where(where_expr).order_by(Acer.c.id.desc())
                    )
                    candidate_rows = result_proxy.mappings().all()
                print(f"[Vector AI Search] Fetched {len(candidate_rows)} candidate rows with embeddings.")

                # --- 6. Score candidates using cosine similarity ---
                scored_candidates = []
                for row in candidate_rows:
                    db_embedding_data = row.get(embed_col_name)
                    if db_embedding_data is None:
                        continue # Defensive check
                    # Parse the stored embedding (JSON string -> list of floats)
                    db_vec = _parse_vec(db_embedding_data)
                    if db_vec is None:
                        continue # Skip rows with unparsable embeddings
                    # Calculate similarity
                    similarity_score = _cosine(q_vec, db_vec)
                    # --- CHANGE 2 Core: Apply threshold to filter relevance ---
                    # Only add candidates that meet or exceed the minimum similarity.
                    # If VEC_ONLY_MIN is too low, increase it (e.g., from 0.15 to 0.20 or 0.25).
                    # Ensure VEC_ONLY_MIN is defined in your config (e.g., at the top or via env var).
                    if similarity_score >= VEC_ONLY_MIN:
                        row_dict = dict(row)
                        row_dict["_score"] = float(similarity_score)
                        scored_candidates.append(row_dict)
                print(f"[Vector AI Search] {len(scored_candidates)} candidates passed similarity threshold ({VEC_ONLY_MIN}).")

                # --- 7. Sort candidates by similarity score descending ---
                scored_candidates.sort(key=lambda x: x["_score"], reverse=True)

                # --- 8. Calculate the total number of results BEFORE pagination ---
                total_count = len(scored_candidates)

                # --- CHANGE 2 Final Check & CHANGE 3: Apply pagination ---
                # If no scored candidates, results remain empty, preventing random display.
                if total_count > 0:
                    # Apply pagination (offset and limit) to get the final page of results
                    paginated_results = scored_candidates[offset : offset + per_page] # Use updated per_page
                    print(f"[Vector AI Search] Returning {len(paginated_results)} results (offset={offset}, limit={per_page}). Total available: {total_count}")

                    # Prepare data for template
                    results = [ {k:v for k,v in r.items() if k != "_score"} for r in paginated_results ]
                    total = total_count # Use the total count for correct pagination
                    pages = max(1, (total // per_page) + (1 if total % per_page else 0))
                else:
                    print(f"[Vector AI Search] No candidates met the similarity threshold ({VEC_ONLY_MIN}). Returning no results.")
                    results = []
                    total = 0
                    pages = 1 # Or could be 0, depending on template logic

        # Set q_display to ai_q so the template knows it's an AI search mode
        q_display = ai_q

    # ---------- Standard search (unchanged) ----------
    elif q_raw or any([sel_industries, sel_sections, sel_rules, sel_parties, sel_s1, sel_s2]):
        # Reset per_page to standard if not AI search (redundant, but clear)
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
            # First: NLQ
            where_expr = and_(*(base_filters + [ _nlq_clause(q_raw) ])) if base_filters else _nlq_clause(q_raw)
            total, results = _run_expr(where_expr)
            # Fallback: plain LIKE if NLQ found 0
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

    # Pagination links (mostly unchanged, but uses potentially updated per_page logic)
    def page_url(n):
        params = {
            'industry': sel_industries, 'section': sel_sections, 'rule': sel_rules,
            'party': sel_parties, 's1': sel_s1, 's2': sel_s2, 'page': n
        }
        if is_ai_search and ai_q: # Pass ai_search and ai_q for AI search pagination
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

    # Pass the separate AI text parts and potential error to the template
    return render_template(
        "index.html",
        industries=industries, sections=sections, rules=rules, parties=parties, subjects=subjects,
        selected={"industry": set(sel_industries), "section": set(sel_sections),
                  "rule": set(sel_rules), "party": set(sel_parties),
                  "s1": set(sel_s1), "s2": set(sel_s2)},
        q=q_display,
        results=results, page=page, pages=pages, total=total,
        page_urls=page_urls, prev_url=prev_url, next_url=next_url,
        # Pass the new separate AI explanation parts
        ai_statute_text=ai_statute_text,
        ai_circular_text=ai_circular_text,
        ai_error=ai_error, # Pass any AI error for display
        ai_q=ai_q # Keep passing ai_q if needed by template logic
    )

# ... (rest of the code like /case/<int:rid>, /summarize_results, admin routes, etc. remains unchanged) ...


# --------- Public case detail ---------
@app.get("/case/<int:rid>")
def public_case_detail(rid: int):
    with ENGINE.connect() as conn:
        row = conn.execute(select(Acer).where(Acer.c.id == rid)).mappings().first()
    if not row: abort(404)
    r = dict(row)
    head_plain = _strip_all_html(r.get("summary_head_note") or "")
    headnotes_html = _bold_held_and_prefix(head_plain)
    qa_plain   = _strip_all_html(r.get("question_answered") or "")
    qa_items   = _split_points(qa_plain)
    citations_text = _strip_all_html(r.get("citation") or "")
    full_case_html = r.get("full_case_law") or ""
    args = request.args
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
    if q:              clauses.append(_nlq_clause(q))
    where_expr = and_(*clauses) if clauses else text("1=1")
    with ENGINE.connect() as conn:
        rows = conn.execute(
            select(Acer.c.id, Acer.c.case_law_number, Acer.c.name_of_party, Acer.c.date)
            .where(where_expr).order_by(Acer.c.id.desc())
        ).mappings().all()
    list_results = [dict(x) for x in rows]
    return render_template(
        "public_case_detail.html",
        r=r, headnotes_html=headnotes_html, full_case_html=full_case_html,
        citations_text=citations_text, qa_items=qa_items,
        search_summary=" | ".join(filter(None, [
            sel_industries and ("Industry: " + ", ".join(sel_industries)),
            sel_sections   and ("Section: " + ", ".join(sel_sections)),
            sel_rules      and ("Rule: " + ", ".join(sel_rules)),
            sel_parties    and ("Party: " + ", ".join(sel_parties)),
            sel_s1         and ("Subject 1: " + ", ".join(sel_s1)),
            sel_s2         and ("Subject 2: " + ", ".join(sel_s2)),
            q and f'Search: "{q}"'
        ])),
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
from sqlalchemy import func, or_  # Make sure func and or_ are imported

# ... other imports ...



@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Login using ADMIN_USERS (email -> sha256(password) hash).
    Expects form fields: name="email", name="password".
    On success, sets session['admin_email'] and session['last_seen_at'].
    Honors ?next=... redirect if present.
    """
    error = None
    if request.method == 'POST':
        email = (request.form.get('email') or request.form.get('username') or '').strip().lower()
        password = request.form.get('password') or ''

        # Compute SHA-256 using your helper if present, else fallback
        try:
            computed_hash = hash_password(password)  # your existing helper
        except NameError:
            import hashlib
            computed_hash = hashlib.sha256(password.encode()).hexdigest()

        stored_hash = ADMIN_USERS.get(email)

        if stored_hash and stored_hash == computed_hash:
            # success: establish session and redirect
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
    """
    Clear admin session and return to the login page.
    """
    session.pop('admin_email', None)
    session.pop('last_seen_at', None)
    return redirect(url_for('login'))

@app.get("/admin")
def admin_root():
    return redirect(url_for("admin_caselaws"))

@app.get("/admin/caselaws")
def admin_caselaws():
    q = (request.args.get("q") or "").strip()
    page = max(int(request.args.get("page", 1)), 1)
    per_page = min(max(int(request.args.get("per_page", 20)), 1), 200)
    offset = (page - 1) * per_page
    mode = request.args.get("mode", "edit") # Get mode for edit/delete panels
    edit_item = None # Initialize edit_item

    # Check if we are editing a specific item
    if 'rid' in request.args or (request.referrer and 'rid=' in request.referrer):
        # This part might need refinement based on how you handle edits
        # For now, assuming edit_item is handled elsewhere or passed differently
        # If you fetch edit_item based on 'rid', do it here
        pass

    where_clause = _nlq_clause(q) if q else text("1=1")

    with ENGINE.connect() as conn:
        # --- Existing Queries ---
        total = conn.execute(select(func.count()).select_from(Acer).where(where_clause)).scalar_one()
        rows = conn.execute(
            select(Acer.c.id, Acer.c.case_law_number, Acer.c.name_of_party, Acer.c.date, Acer.c.state)
            .where(where_clause).order_by(Acer.c.id.desc()).limit(per_page).offset(offset)
        ).mappings().all()

        # --- New Statistics Queries ---
        # 1. Total Case Laws (overall, not filtered)
        total_case_laws = conn.execute(select(func.count()).select_from(Acer)).scalar_one()

        # 2. AAR Case Laws - Count rows containing 'aar' but NOT containing 'aaar'
        #    Also ensure it's case-insensitive
        aar_case_laws = conn.execute(
            select(func.count())
            .select_from(Acer)
            .where(
                and_(
                    func.lower(Acer.c.type_of_court).contains('aar'),      # Must contain 'aar'
                    ~func.lower(Acer.c.type_of_court).contains('aaar')     # Must NOT contain 'aaar' (~ is NOT)
                )
            )
        ).scalar_one()

        # 3. AAAR Case Laws - Count rows containing 'aaar' (case-insensitive)
        #    This one is fine as it specifically looks for 'aaar'
        aaar_case_laws = conn.execute(
            select(func.count())
            .select_from(Acer)
            .where(func.lower(Acer.c.type_of_court).contains('aaar'))
        ).scalar_one()


        # 4. High Court Case Laws - Adjust the filter condition if needed
        high_court_case_laws = conn.execute(
            select(func.count())
            .select_from(Acer)
            .where(or_(
                func.lower(Acer.c.type_of_court).contains('hc'),
                func.lower(Acer.c.type_of_court).contains('high-court')
            ))
        ).scalar_one()

        # 5. Supreme Court Case Laws - Adjust the filter condition if needed
        supreme_court_case_laws = conn.execute(
            select(func.count())
            .select_from(Acer)
            .where(or_(
                func.lower(Acer.c.type_of_court).contains('supreme court'),
                func.lower(Acer.c.type_of_court).contains('supreme-court')
            ))
        ).scalar_one()

        # Handle potential edit item (you might need to adjust how rid is obtained)
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
        # --- Existing Variables ---
        rows=rows,
        page=page,
        per_page=per_page,
        total=total,
        q=q,
        table_name=TABLE_NAME,
        mode=mode, # Pass mode to template
        edit_item=edit_item, # Pass edit_item to template
        # --- New Statistics Variables ---
        total_case_laws=total_case_laws,
        aar_case_laws=aar_case_laws,
        aaar_case_laws=aaar_case_laws,
        high_court_case_laws=high_court_case_laws,
        supreme_court_case_laws=supreme_court_case_laws
    )

# ... other routes ...

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
        # --- Potential Fix: Add validation/checks ---
        errors = []
        for col in ALL_COLS:
            if col == "id": continue
            # Get the value, defaulting to None or empty string if not found
            # request.form.get(col) returns None if key is missing, but the 'if col in request.form' prevents that path.
            # It returns the actual value (possibly empty string "") if key exists.
            value = request.form.get(col)

            # Example validation: Check for required fields (adjust the list as needed)
            # Replace ['case_law_number', 'name_of_party'] with your actual required columns
            required_fields = ['case_law_number', 'name_of_party'] # Example
            if col in required_fields and not value:
                 errors.append(f"Field '{col}' is required.")
                 # Or handle it differently, like setting a default
                 # data[col] = "Unknown" # Example default

            # Example: Handle potential type conversion for 'year' if needed explicitly
            # (Database driver often handles this, but explicit check can help)
            # if col == 'year' and value:
            #     try:
            #         data[col] = int(value)
            #     except ValueError:
            #         errors.append(f"Invalid year format for '{col}': {value}")
            #         # Decide how to handle invalid data - maybe skip, set default, or return error
            # elif col == 'year' and not value:
            #      # Decide if NULL/None is acceptable for year in DB, or set default
            #      # data[col] = None # Or some default like datetime.now().year if appropriate
            #      pass # Let it be empty/None if DB allows

            # Only add non-empty values, or decide based on your DB schema
            # If DB allows NULL, you might want to add even empty values
            # If DB requires a value, ensure it's present before this step
            # This current logic adds the key if it was in the form, regardless of value content.
            # You might need to refine this.
            # Example: Only add if value is not None and not an empty string
            # if value is not None and value != "":
            #     data[col] = value
            # Or, add it regardless (current logic, might cause issues if DB field is NOT NULL and value is empty string)
            else:
                 data[col] = value

        # --- End Potential Fix ---

        # --- Check for validation errors ---
        if errors:
            # Flash errors and re-render the form
            for error in errors:
                flash(error, "err")
            # Pass the submitted data back to the template to avoid re-typing everything
            return render_template("admin_edit.html", r=data, mode="new"), 400 # 400 Bad Request might be appropriate

        # --- Proceed with insertion if no validation errors ---
        try: # Add a try-except block to catch database errors
            with ENGINE.begin() as conn:
                conn.execute(insert(Acer).values(**data))
            flash("New case added.", "ok")
            return redirect(url_for("admin_caselaws"))
        except Exception as e: # Catch database errors
            # Log the error (print to console or use logging module)
            print(f"Database Insert Error: {e}")
            # Flash a user-friendly message
            flash(f"Error saving case: {str(e)}", "err") # Or a generic message
            # Re-render the form with the submitted data
            return render_template("admin_edit.html", r=data, mode="new"), 500 # 500 Internal Server Error

    # Handle GET request (show the form)
    empty = {c: "" for c in ALL_COLS}
    empty["id"] = ""
    return render_template("admin_edit.html", r=empty, mode="new")

# --- ADD THIS FUNCTION for HTML Cleaning ---
def clean_html(raw_html: str) -> str:
    """
    Removes HTML tags and converts common HTML entities for clean text export.
    """
    if not isinstance(raw_html, str):
        return str(raw_html) if raw_html is not None else ""
    # Remove HTML tags using regex
    clean_text = re.sub(r'<[^>]*>', '', raw_html)
    # Convert common HTML entities
    clean_text = clean_text.replace('&nbsp;', ' ')
    clean_text = clean_text.replace('&amp;', '&')
    clean_text = clean_text.replace('<', '<')
    clean_text = clean_text.replace('>', '>')
    clean_text = clean_text.replace('&quot;', '"')
    clean_text = clean_text.replace('&#39;', "'")
    # Add more entity replacements if needed
    return clean_text.strip()

# --- ADD THIS FUNCTION for HTML Cleaning (Enhanced) ---
def clean_html(raw_html: str) -> str:
    """
    Removes HTML tags and converts common HTML entities for clean text export.
    This function specifically handles <p>, <ul>, <li>, and other common tags.
    """
    if not isinstance(raw_html, str):
        return str(raw_html) if raw_html is not None else ""

    # Remove HTML tags using regex
    # This pattern matches any opening or closing tag, including self-closing ones like <br/>
    clean_text = re.sub(r'<[^>]+>', '', raw_html)

    # Convert common HTML entities
    clean_text = clean_text.replace('&nbsp;', ' ')
    clean_text = clean_text.replace('&amp;', '&')
    clean_text = clean_text.replace('<', '<')
    clean_text = clean_text.replace('>', '>')
    clean_text = clean_text.replace('&quot;', '"')
    clean_text = clean_text.replace('&#39;', "'")

    # Clean up extra whitespace (optional, makes output look neater)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    return clean_text

# --- Define the desired column order for export ---
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
    'citation', # Included in export but cleaned
    'basic_detail',
    'summary_head_note',
    'question_answered'
    # 'full_case_law' is intentionally excluded
]

@app.get("/admin/export_csv")
def admin_export_csv():
    try:
        # --- 1. Check if required objects are defined ---
        # (Add checks similar to the previous example if needed,
        # or ensure these are imported/globally accessible)
        # global ENGINE, Acer, ALL_COLS # Example for global access

        if 'ALL_COLS' not in globals() or not ALL_COLS:
            print("Error: ALL_COLS is not defined or is empty.")
            return "Error: Column list (ALL_COLS) is missing or empty.", 500

        if 'Acer' not in globals() or Acer is None:
            print("Error: Table object 'Acer' is not defined.")
            return "Error: Table object 'Acer' is missing.", 500

        if 'ENGINE' not in globals() or ENGINE is None:
            print("Error: Database engine 'ENGINE' is not defined.")
            return "Error: Database engine is missing.", 500

        # --- 2. Build column list based on EXPORT_COLUMN_ORDER ---
        # Ensure the column names in EXPORT_COLUMN_ORDER exist in the table
        cols_to_select = [getattr(Acer.c, c) for c in EXPORT_COLUMN_ORDER if hasattr(Acer.c, c)]

        selected_column_names = [c.name for c in cols_to_select]

        if not cols_to_select:
            print("Warning: No valid columns found based on EXPORT_COLUMN_ORDER.")
            df = pd.DataFrame(columns=EXPORT_COLUMN_ORDER)
        else:
            # --- 3. Execute Query ---
            with ENGINE.connect() as conn:
                result = conn.execute(select(*cols_to_select).order_by(Acer.c.id.desc()))
                rows = result.mappings().all()

            # --- 4. Create DataFrame ---
            df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=selected_column_names)

            # --- 5. Clean HTML Columns ONLY for Export ---
            print(f"Fetched {len(rows)} rows. Cleaning HTML for export...")

            # Define which columns contain HTML that needs cleaning for export
            html_columns_to_clean = [
                'summary_head_note',
                'question_answered',
                'basic_detail',
                'citation' # Add 'citation' to the list of columns to clean
            ]

            # --- IMPORTANT: Ensure columns exist in DataFrame before trying to clean ---
            existing_html_columns = [col for col in html_columns_to_clean if col in df.columns]

            if existing_html_columns:
                 # Apply the clean_html function to each specified column
                 for col_name in existing_html_columns:
                     # Fill NaN/None with empty string temporarily for cleaning
                     df[col_name] = df[col_name].fillna("").apply(lambda x: clean_html(x) if pd.notna(x) else "")

            # --- 6. Reorder DataFrame Columns ---
            final_column_order = [col for col in EXPORT_COLUMN_ORDER if col in df.columns]
            if final_column_order:
                df = df[final_column_order]

            print("HTML cleaning and column reordering for export completed.")

        # --- 7. Write to Buffer ---
        buf = io.BytesIO()
        # Excel-friendly: UTF-8 with BOM
        df.to_csv(buf, index=False, encoding="utf-8-sig")
        buf.seek(0)

        # --- 8. Send File ---
        # For Flask 2.0+
        return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="cases_cleaned_export.csv")

    except Exception as e:
        # --- 9. Handle Unexpected Errors ---
        print(f"An error occurred during CSV export: {e}")
        import traceback
        traceback.print_exc()
        return f"Internal Server Error during export: {str(e)}", 500

# ---- Aliases so "Bulk Download" buttons never 404 ----
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
    # --- Handle 'date' column specifically ---
    if "date" in df.columns:
        original_date_dtype = df["date"].dtype
        print(f"[DEBUG] Original 'date' column dtype: {original_date_dtype}")
        print(f"[DEBUG] Sample 'date' values before coercion:\n{df['date'].head(10)}")

        # --- Attempt explicit format parsing ---
        # --- IMPORTANT: CHANGE THE FORMAT STRING BELOW TO MATCH YOUR EXCEL FILE ---
        # Common formats: DD-MM-YYYY -> "%d-%m-%Y", MM/DD/YYYY -> "%m/%d/%Y", YYYY-MM-DD -> "%Y-%m-%d"
        # You MUST identify the format used in your specific Excel file.
        excel_date_format = "%d-%m-%Y" # <--- CHANGE THIS LINE ---

        if excel_date_format:
            try:
                # Parse using the specific format first for accuracy
                df["date"] = pd.to_datetime(df["date"], format=excel_date_format, errors="coerce")
                print(f"[INFO] Parsed dates using explicit format '{excel_date_format}'.")
            except (ValueError, TypeError) as e_format:
                # If the specific format fails, fall back to default parsing and warn
                print(f"[WARNING] Failed to parse all dates with format '{excel_date_format}': {e_format}. Falling back to default parser.")
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            # If no format is specified (excel_date_format is empty/falsy), use default parsing
            print("[INFO] No specific date format provided, using default pd.to_datetime parser.")
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        print(f"[DEBUG] 'date' column dtype after pd.to_datetime: {df['date'].dtype}")
        print(f"[DEBUG] Sample 'date' values after pd.to_datetime:\n{df['date'].head(10)}")
        nat_count_after_to_datetime = df["date"].isna().sum()
        print(f"[DEBUG] Number of NaT/NaN in 'date' after pd.to_datetime: {nat_count_after_to_datetime}")

        # --- CRITICAL: Explicitly convert pd.NaT to None ---
        # This step ensures that SQLAlchemy/PyMySQL receives None, not pd.NaT.
        nat_count_before_apply = df["date"].isna().sum()
        # Use apply with a lambda to check for pd.isna (which correctly identifies NaT) and replace
        df["date"] = df["date"].apply(lambda x: None if pd.isna(x) else x)
        nat_count_after_apply = df["date"].isna().sum() # Should be 0 for the date column now
        print(f"[INFO] Date column NaT conversion: {nat_count_before_apply} NaT/NaN values found, {nat_count_before_apply - nat_count_after_apply} converted to None.")
        print(f"[DEBUG] 'date' column dtype after NaT->None conversion: {df['date'].dtype}")
        print(f"[DEBUG] Sample 'date' values after NaT->None conversion:\n{df['date'].head(10)}")

    # --- Handle 'year' column ---
    if "year" in df.columns:
        def _to_year(v):
            # Handle cases where year might derive from an invalid date (which is now None)
            # or is itself invalid/missing.
            if pd.isna(v): # This correctly handles None, np.nan, pd.NaT
                return None
            try:
                s = str(v).strip()
                # Handle potential float representations like 2023.0
                if '.' in s and s.replace('.', '', 1).isdigit():
                     # Check if it's a whole number float
                     float_val = float(s)
                     if float_val.is_integer():
                         s = str(int(float_val)) # Convert 2023.0 to "2023"
                if len(s) >= 4 and s[:4].isdigit():
                    return int(s[:4])
                # If it's a shorter number or non-standard, try converting float then int
                # This handles cases like "23" -> 2023 (if that logic is intended, otherwise remove)
                # For now, let's stick to 4-digit extraction or float conversion
                return int(float(s)) # This will raise ValueError if s is not numeric
            except (ValueError, TypeError): # Catch conversion errors explicitly
                # print(f"[DEBUG] Could not convert year value '{v}' to integer.")
                return None
        df["year"] = df["year"].map(_to_year)
        # Optional: If year was missing but date is valid, derive year from date
        # Use pd.notna for boolean mask on datetime series if 'date' column exists and was processed
        if "date" in df.columns:
             # Fill missing years where date is available and successfully parsed (not None)
             mask_years_missing_dates_valid = df["year"].isna() & df["date"].notna()
             if mask_years_missing_dates_valid.any():
                 print(f"[INFO] Deriving {mask_years_missing_dates_valid.sum()} year(s) from parsed 'date' column.")
                 df.loc[mask_years_missing_dates_valid, "year"] = df.loc[mask_years_missing_dates_valid, "date"].dt.year

    # --- Handle other object columns (strings etc.) ---
    for c in df.columns:
        # Apply string conversion and stripping only to object columns
        if df[c].dtype == object:
            # Use pd.isna to correctly check for None, NaN, NaT in object columns
            # Map None/NaN values to None, others to stripped string
            df[c] = df[c].map(lambda x: str(x).strip() if not pd.isna(x) else None)
            # Debug: Check if any 'NaT' strings slipped through (shouldn't happen after datetime conversion)
            if c == 'date' and df[c].isin(['NaT']).any():
                 print(f"[ERROR] Found string 'NaT' in 'date' column after object processing! This indicates a problem.")
                 # Force conversion again if needed (defensive)
                 df[c] = df[c].replace('NaT', None)

    # Debugging: Print final dtypes if needed
    # print("[DEBUG] Final dtypes after _coerce_types:")
    # print(df.dtypes)
    # print("[DEBUG] Head of DataFrame after _coerce_types:")
    # print(df.head())
    return df

@app.get("/admin/bulk_upload/template")
def admin_bulk_template():
    df = pd.DataFrame(columns=_TEMPLATE_COLUMNS)
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    buf.seek(0)
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="bulk_upload_template.csv")

# alias for older templates pointing to /admin/bulk_template
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

    # --- CRUCIAL FIX: Explicitly replace nan/NaN values with None ---
    # This ensures that pandas' NaN (which can be numpy.nan) is converted
    # to Python None, which SQLAlchemy/PyMySQL correctly handles as SQL NULL.
    # Place this immediately after reading the DataFrame and before any processing.
    df = df.replace({np.nan: None})
    # --- End of Fix ---

    # Normalize column names: clean and map using _SYNONYMS
    def clean_key(s):
        return re.sub(r"\s+", " ", s.strip().lower())

    col_mapping = {}
    synonym_keys = {clean_key(k): v for k, v in _SYNONYMS.items()}

    for col in df.columns:
        c = clean_key(col)
        if c in synonym_keys:
            col_mapping[col] = synonym_keys[c]
        elif c.replace(" ", "_") in _TEMPLATE_COLUMNS:
            # Direct match via snake_case
            mapped = c.replace(" ", "_")
            if mapped in _TEMPLATE_COLUMNS:
                col_mapping[col] = mapped
        elif c in [x.lower() for x in _TEMPLATE_COLUMNS]:
            # Exact case-insensitive match
            orig = [x for x in _TEMPLATE_COLUMNS if x.lower() == c][0]
            col_mapping[col] = orig

    if not col_mapping:
        flash("No valid columns found. Please use correct column names.", "err")
        return redirect(url_for("admin_caselaws"))

    df = df.rename(columns=col_mapping)

    # Keep only columns present in database
    df = df[[c for c in _TEMPLATE_COLUMNS if c in df.columns]]

    # Coerce types (ensure this doesn't reintroduce problematic NaNs)
    df = _coerce_types(df)
    # Apply the replace again after type coercion as a safeguard
    df = df.replace({np.nan: None})

    # Convert to records (list of dictionaries)
    # The replace({np.nan: None}) should have handled this, but using where/notnull is an alternative
    # records = df.where(pd.notnull(df), None).to_dict("records")
    # Let's use the safer approach with replace ensuring None
    records = df.to_dict("records")
    # The df.replace({np.nan: None}) call above should have already converted NaNs to None in the df,
    # and thus in the dictionaries created by to_dict("records").

    if not records:
        flash("No data to insert.", "err")
        return redirect(url_for("admin_caselaws"))

    try:
        with ENGINE.begin() as conn: # Assuming ENGINE is your SQLAlchemy engine
            # Delete existing rows with same case_law_number
            case_numbers = [r["case_law_number"] for r in records if r["case_law_number"] is not None]
            if case_numbers:
                # Assuming Acer is your table object
                conn.execute(delete(Acer).where(Acer.c.case_law_number.in_(case_numbers)))
            # Insert new data
            conn.execute(insert(Acer), records) # Assuming Acer is your table object
    except Exception as e:
        # It's good practice to log the full error for debugging
        print(f"Database insert failed: {e}") # Add this for server logs
        flash(f"Database insert failed: {e}", "err")
        return redirect(url_for("admin_caselaws"))

    flash(f"Successfully uploaded {len(records)} row(s). Duplicates were overwritten.", "ok")
    return redirect(url_for("admin_caselaws"))
# ======================= Contact =======================
@app.route("/contact", methods=["GET", "POST"])
def contact():
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
    # TEMPORARY: For debugging only. Remove after fixing.
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


if __name__ == "__main__":
    app.run(debug=True)
