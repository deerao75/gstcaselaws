from flask import Flask, request, render_template, abort, redirect, url_for, flash, send_file, jsonify, render_template_string
from sqlalchemy import create_engine, MetaData, Table, select, or_, func, text, and_, insert, update, delete
from bs4 import BeautifulSoup
import os, re, io, ssl, json, time, hashlib
import pandas as pd
from collections import defaultdict
from datetime import date, datetime

# ======================= DB CONFIG =======================
# (Render/TiDB-safe: no hardcoded Windows CA path; supports Secret Files)
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret")

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

# Optional override (0/1/true/false). If not set, infer from port/host.
DB_USE_TLS = os.getenv("DB_USE_TLS")
if DB_USE_TLS is not None:
    use_tls = DB_USE_TLS.strip().lower() in ("1", "true", "yes")
else:
    use_tls = (DB_PORT == "4000") or ("tidbcloud.com" in (DB_HOST or "").lower())

if use_tls:
    # Show what's actually mounted in /etc/secrets (Render mounts Secret Files here)
    try:
        print("[TLS] /etc/secrets contents:", os.listdir("/etc/secrets"))
    except Exception as _e:
        print("[TLS] Could not list /etc/secrets:", _e)

    # Candidate CA paths (env + common filenames)
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
        # Fall back to system CAs; many managed DBs are signed by public roots.
        try:
            ctx = ssl.create_default_context()
            CONNECT_ARGS = {"ssl": ctx}
            print("[DB] TLS enabled with system CA bundle (no explicit CA file found).")
        except Exception as e:
            # Last resort: error out with helpful message
            raise RuntimeError(
                "TLS required but no CA file found and system bundle failed. "
                "Set env TIDB_SSL_CA to /etc/secrets/<your-secret-file> or add a Secret File."
            ) from e
else:
    print("[DB] TLS not required (DB_USE_TLS=0 or non-TiDB settings).")

from sqlalchemy import create_engine
ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=CONNECT_ARGS)
metadata = MetaData()
Acer = Table(TABLE_NAME, metadata, autoload_with=ENGINE)

ALL_COLS = [
    "id","case_law_number","name_of_party","state","year","type_of_court",
    "industry_sector","subject_matter1","subject_matter2","section","rule",
    "case_name","date","basic_detail","summary_head_note","citation",
    "question_answered","full_case_law",
]
SEARCHABLE_COLS = [
    "case_law_number","name_of_party","case_name","state",
    "section","rule","citation","industry_sector",
    "summary_head_note"#,"full_case_law"#
]

# ======================= OPENAI CONFIG =======================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_env_model = (os.getenv("OPENAI_MODEL") or "").strip()
_ALLOWED = {"gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"}
OPENAI_MODEL = _env_model if _env_model in _ALLOWED else "gpt-4o-mini"

def _openai_chat(messages, max_tokens=800, temperature=0.4):
    """
    Version-compatible OpenAI chat call:
    - Try new client (`from openai import OpenAI`) first
    - Fall back to legacy `openai.ChatCompletion.create`
    Raises Exception with clear text so caller can show it to the user.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

    try:
        # New SDK (openai>=1.0)
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e_new:
        try:
            # Legacy SDK
            import openai  # type: ignore
            openai.api_key = OPENAI_API_KEY
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (resp["choices"][0]["message"]["content"] or "").strip()
        except Exception as e_old:
            raise RuntimeError(f"OpenAI call failed: {e_old or e_new}")

def _openai_chat_json(messages, model=None, max_tokens=900, temperature=0.3):
    """
    JSON-mode helper for Chat Completions.
    Returns a dict. Falls back to legacy SDK and tries to parse JSON.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
    model = model or OPENAI_MODEL or "gpt-4o-mini"

    # New SDK path
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception:
        # Legacy fallback: instruct JSON and parse
        import openai  # type: ignore
        openai.api_key = OPENAI_API_KEY
        _msgs = messages + [{"role": "system", "content": "Return a single JSON object as your entire reply."}]
        resp = openai.ChatCompletion.create(
            model=model,
            messages=_msgs,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        raw = resp["choices"][0]["message"]["content"] or "{}"
        try:
            return json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, flags=re.S)
            return json.loads(m.group(0)) if m else {}

# ======================= RUNTIME CACHES (speed-ups) =======================
_AI_CACHE = {}     # { key: {"value": str, "exp": epoch_seconds} }
_META_CACHE = {}   # { name: {"value": any, "exp": epoch_seconds} }

def _ai_cache_get(key: str):
    e = _AI_CACHE.get(key)
    if not e: return None
    if e["exp"] < time.time():
        _AI_CACHE.pop(key, None)
        return None
    return e["value"]

def _ai_cache_set(key: str, value: str, ttl_sec: int = 900):
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

# ======================= HELPERS =======================
def like_filters(q: str):
    terms = []
    for col in SEARCHABLE_COLS:
        if hasattr(Acer.c, col):
            terms.append(func.lower(getattr(Acer.c, col)).like(f"%{q.lower()}%"))
    return or_(*terms) if terms else text("1=1")

# ---- Natural Language Query helpers ----
_STOPWORDS = {
    "a","an","the","this","that","these","those","me","we","us","you","your","yours",
    "give","show","find","get","provide","list","tell","display","search","look","fetch",
    "please","kindly","need","want","would","like","to","for","on","about","regarding","related","with","in","of","by","under","and","or",
    "cases","case","laws","law","caselaw","caselaws"
}
_KEEP_SHORT = {"itc","gst","rcm","hsn","pos","b2b","b2c"}
_DOMAIN_PHRASES = [
    "composition scheme","input tax credit","place of supply","reverse charge",
    "works contract","advance ruling","e way bill","e-way bill","zero rated",
    "export of services","intermediary services","blocked credit","time of supply",
    "credit note","debit note","classification dispute","valuation","anti profiteering",
    "refund of itc","job work","pure agent"
]

def _nlq_keywords(q: str):
    """
    Extract meaningful keywords/phrases from a natural-language query.
    - lowers, removes punctuation (keeps hyphens), strips stopwords
    - preserves domain phrases as single tokens if present
    - returns deduped list in the original order of appearance
    """
    if not q: return []
    s = q.lower()
    # soft normalize punctuation (keep hyphen so 'e-way' works)
    s = re.sub(r"[^\w\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    kept = []
    seen = set()

    # detect domain phrases first (longest wins)
    s_for_phrase = " " + s + " "
    for ph in sorted(_DOMAIN_PHRASES, key=len, reverse=True):
        ph_norm = " " + ph + " "
        if ph_norm in s_for_phrase:
            if ph not in seen:
                kept.append(ph)
                seen.add(ph)
            # remove the phrase so its words don't reappear as singles
            s_for_phrase = s_for_phrase.replace(ph_norm, " ")
    # residual singles
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
    """
    Build a SQLAlchemy clause that supports natural language:
    AND over tokens, where each token is ORed over SEARCHABLE_COLS.
    Fallback to simple like_filters if tokenization yields nothing.
    """
    toks = _nlq_keywords(query)
    if not toks:
        return like_filters(query)
    return and_(*[like_filters(t) for t in toks])

def _distinct(colname):
    # 5-minute cache to avoid repeated DB hits for dropdowns
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
    # 5-minute cache for subjects tree
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

def _strip_all_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = soup.get_text(separator="\n")
    text = re.sub(r"\r\n?","\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
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

# --------- Formatting helpers for summarization ----------
def _selected_topic(args) -> str:
    """Compute the main filter topic to guide the summary."""
    s1 = args.getlist("s1")
    s2 = args.getlist("s2")
    sections = args.getlist("section")
    rules = args.getlist("rule")
    inds = args.getlist("industry")
    is_ai = args.get("ai_search") == "true"
    ai_q = (args.get("ai_q") or "").strip() if is_ai else ""
    q = (args.get("q") or "").strip()

    # Priority: Subject-2 > Subject-1 > Section > Rule > Industry > ai_q > q
    if s2:   return ", ".join(s2)
    if s1:   return ", ".join(s1)
    if sections: return "Section(s): " + ", ".join(sections)
    if rules:    return "Rule(s): " + ", ".join(rules)
    if inds:     return "Industry: " + ", ".join(inds)
    if ai_q:     return ai_q
    if q:        return q
    return "GST case law"

def _clean_summary(text: str) -> str:
    """Remove star/bullet markers; drop generic wrap-ups; normalize paragraphs."""
    if not text:
        return ""
    # strip bullets/stars at line starts
    lines = []
    for ln in text.splitlines():
        ln = re.sub(r'^\s*[\*\-•]+\s*', '', ln)          # *, -, •
        ln = re.sub(r'^\s*\d+\.\s*', '', ln)             # numbered bullets
        lines.append(ln)
    text = "\n".join(lines)

    # Remove generic concluding lines
    text = re.sub(r'(?im)^\s*(overall|in summary|in conclusion|to conclude|in short)\b.*$', '', text)

    # Collapse 3+ blank lines -> single blank line
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text

# ======================= AI EXPLANATION (FASTER + CACHED) =======================
def get_ai_explanation(prompt_text):
    """
    Faster, cached AI step for 'Search with AI'.
    - 1–2 short paragraphs + final 'Keywords:' line (5–10 comma-separated).
    - Cached for 15 minutes per (query + selected filters).
    """
    args = request.args
    key_src = "|".join([
        "ai", (prompt_text or "").strip(),
        ",".join(sorted(args.getlist("industry"))),
        ",".join(sorted(args.getlist("section"))),
        ",".join(sorted(args.getlist("rule"))),
        ",".join(sorted(args.getlist("party"))),
        ",".join(sorted(args.getlist("s1"))),
        ",".join(sorted(args.getlist("s2"))),
    ])
    cache_key = "ai_explain:" + hashlib.sha256(key_src.encode("utf-8")).hexdigest()
    cached = _ai_cache_get(cache_key)
    if cached is not None:
        return {"text": cached, "cached": True}

    full_prompt = (
        "You are an expert on Indian GST laws. In 1–2 short paragraphs, answer the query. "
        "Then, on a final line that starts with 'Keywords:', list 5–10 search keywords "
        "(comma-separated). Keep total under ~250 words.\n\n"
        f"Query: {(prompt_text or '').strip()}"
    )
    try:
        text = _openai_chat(
            [
                {"role": "system", "content": "You are a helpful assistant knowledgeable about Indian GST laws."},
                {"role": "user", "content": full_prompt},
            ],
            max_tokens=450,   # smaller = faster
            temperature=0.3,
        )
        _ai_cache_set(cache_key, text, ttl_sec=900)  # 15 minutes
        return {"text": text}
    except Exception as e:
        print("AI explain error:", e)
        return {"error": f"{e}"}

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
    q              = (request.args.get("q") or "").strip()

    results = []
    total = 0
    pages = 1
    ai_response_data = None

    COMMON_COLS = [
        Acer.c.id, Acer.c.case_law_number, Acer.c.name_of_party, Acer.c.date,
        Acer.c.summary_head_note,
        Acer.c.type_of_court, Acer.c.state, Acer.c.year,
        Acer.c.section, Acer.c.rule, Acer.c.citation,
        Acer.c.subject_matter1, Acer.c.subject_matter2,
    ]

    # AI search
    if is_ai_search and ai_q:
        ai_response_data = get_ai_explanation(ai_q)
        keywords_for_search = ai_q
        if ai_response_data and 'text' in ai_response_data:
            lines = ai_response_data['text'].splitlines()
            if lines:
                last = lines[-1]
                if last.lower().startswith("keywords:"):
                    keywords_for_search = last[9:].strip()

        clauses = []
        if sel_industries: clauses.append(Acer.c.industry_sector.in_(sel_industries))
        if sel_sections:   clauses.append(Acer.c.section.in_(sel_sections))
        if sel_rules:      clauses.append(Acer.c.rule.in_(sel_rules))
        if sel_parties:    clauses.append(Acer.c.name_of_party.in_(sel_parties))
        if sel_s1:         clauses.append(Acer.c.subject_matter1.in_(sel_s1))
        if sel_s2:         clauses.append(Acer.c.subject_matter2.in_(sel_s2))
        # NLQ clause instead of plain LIKE (handles "give me the case laws on ...")
        clauses.append(_nlq_clause(keywords_for_search))
        where_expr = and_(*clauses) if clauses else text("1=1")

        page = max(int(request.args.get("page", 1) or 1), 1)
        per_page = 10
        offset = (page - 1) * per_page
        with ENGINE.connect() as conn:
            total = conn.execute(select(func.count()).select_from(Acer).where(where_expr)).scalar_one()
            rows = conn.execute(
                select(*COMMON_COLS).where(where_expr).order_by(Acer.c.id.desc()).limit(per_page).offset(offset)
            ).mappings().all()
        results = [dict(r) for r in rows]
        pages = (total // per_page) + (1 if total % per_page else 0)

    # Standard search
    elif q or any([sel_industries, sel_sections, sel_rules, sel_parties, sel_s1, sel_s2]):
        clauses = []
        if sel_industries: clauses.append(Acer.c.industry_sector.in_(sel_industries))
        if sel_sections:   clauses.append(Acer.c.section.in_(sel_sections))
        if sel_rules:      clauses.append(Acer.c.rule.in_(sel_rules))
        if sel_parties:    clauses.append(Acer.c.name_of_party.in_(sel_parties))
        if sel_s1:         clauses.append(Acer.c.subject_matter1.in_(sel_s1))
        if sel_s2:         clauses.append(Acer.c.subject_matter2.in_(sel_s2))
        if q:              clauses.append(_nlq_clause(q))   # <-- NLQ here too
        where_expr = and_(*clauses) if clauses else text("1=1")

        page = max(int(request.args.get("page", 1) or 1), 1)
        per_page = 10
        offset = (page - 1) * per_page
        with ENGINE.connect() as conn:
            total = conn.execute(select(func.count()).select_from(Acer).where(where_expr)).scalar_one()
            rows = conn.execute(
                select(*COMMON_COLS).where(where_expr).order_by(Acer.c.id.desc()).limit(per_page).offset(offset)
            ).mappings().all()
        results = [dict(r) for r in rows]
        pages = (total // per_page) + (1 if total % per_page else 0)

    # Pagination links
    page = max(int(request.args.get("page", 1) or 1), 1)
    def page_url(n):
        params = {
            'industry': sel_industries, 'section': sel_sections, 'rule': sel_rules,
            'party': sel_parties, 's1': sel_s1, 's2': sel_s2, 'page': n
        }
        if is_ai_search:
            params['ai_search'] = 'true'; params['ai_q'] = ai_q
        else:
            params['q'] = q
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
        q=q, results=results, page=page, pages=pages, total=total,
        page_urls=page_urls, prev_url=prev_url, next_url=next_url,
        ai_response=ai_response_data, ai_q=ai_q
    )

# --------- Public case detail (unchanged logic) ---------
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
    if q:              clauses.append(_nlq_clause(q))  # keep it consistent here too
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

# ======================= SUMMARY HELPERS & ROUTE =======================
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
        clauses.append(_nlq_clause(ai_q))  # NLQ for AI input
    elif q:
        clauses.append(_nlq_clause(q))     # NLQ for normal search
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
    """Turn DB rows into compact docs the model can cite."""
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

def _compose_case_lines(rows, max_chars=220):
    # kept for compatibility; not used by JSON-mode summarizer
    lines = []
    for r in rows:
        case_no = str(r.get("case_law_number") or "").strip()
        party   = str(r.get("name_of_party") or "").strip()

        d = r.get("date")
        if isinstance(d, (date, datetime)):
            date_str = d.strftime("%Y-%m-%d")
        else:
            date_str = str(d or "").strip()

        head = _strip_all_html(r.get("summary_head_note") or "")
        head = re.sub(r"\s+", " ", head).strip()
        if len(head) > max_chars:
            head = head[:max_chars].rstrip() + "…"

        header_bits = [p for p in [case_no, party, date_str] if p]
        header = " — ".join(header_bits) if header_bits else "(case)"
        lines.append(f"- {header}: {head}" if head else f"- {header}")
    return "\n".join(lines)

@app.get("/summarize_results")
def summarize_results():
    try:
        rows = _fetch_rows_for_current_filters(request.args, cap=20)  # 8–20 is a sweet spot
        if not rows:
            return jsonify({"ok": False, "message": "No case law results to summarize for the current filters."})

        topic = _selected_topic(request.args)
        docs  = _build_llm_docs(rows)

        # Build strict, topic-focused prompt; JSON-mode response
        snippets = "\n".join(
            f"- Case: {d['case_no']} — {d['party']} ({d.get('year','')}) | Court: {d.get('court','')} | State: {d.get('state','')}\n"
            f"  Headnote: {d['headnote_clean']}"
            for d in docs
        )
        user_prompt = f"""
Topic: {topic}
Instruction: Write 2–4 tight paragraphs strictly about the Topic. Base only on these headnotes; ignore unrelated material.
Cite 3–6 cases inline as [{{case_no}} — {{party}} ({{year}})]. No bullet points; paragraphs only. No generic wrap-ups.

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
            model="gpt-4o-mini",
            max_tokens=900, temperature=0.3
        )

        answer    = _clean_summary((data or {}).get("answer", ""))
        citations = (data or {}).get("citations", [])

        if not answer.strip():
            return jsonify({"ok": False, "message": "The model returned an empty summary. Try refining filters and retry."})

        # Keep original shape; extra 'citations' is harmless if the front-end ignores it
        return jsonify({"ok": True, "summary": answer, "citations": citations})

    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500

# ======================= ADMIN & CONTACT =======================
@app.get("/admin")
def admin_root():
    return redirect(url_for("admin_caselaws"))

@app.get("/admin/caselaws")
def admin_caselaws():
    q = (request.args.get("q") or "").strip()
    page = max(int(request.args.get("page", 1)), 1)
    per_page = min(max(int(request.args.get("per_page", 20)), 1), 200)
    offset = (page - 1) * per_page
    where_clause = _nlq_clause(q) if q else text("1=1")  # NLQ in admin search too
    with ENGINE.connect() as conn:
        total = conn.execute(select(func.count()).select_from(Acer).where(where_clause)).scalar_one()
        rows = conn.execute(
            select(Acer.c.id, Acer.c.case_law_number, Acer.c.name_of_party, Acer.c.date, Acer.c.state)
            .where(where_clause).order_by(Acer.c.id.desc()).limit(per_page).offset(offset)
        ).mappings().all()
    return render_template(
        "admin_caselaws.html",
        rows=rows, page=page, per_page=per_page, total=total, q=q, table_name=TABLE_NAME
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

@app.get("/admin/export_csv")
def admin_export_csv():
    cols = [getattr(Acer.c, c) for c in ALL_COLS if hasattr(Acer.c, c)]
    with ENGINE.connect() as conn:
        rows = conn.execute(select(*cols).order_by(Acer.c.id.desc())).mappings().all()
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[c.name for c in cols])
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    buf.seek(0)
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="cases.csv")

@app.route("/admin/delete/<int:rid>", methods=["POST"])
def admin_delete_case(rid):
    with ENGINE.begin() as conn:
        conn.execute(delete(Acer).where(Acer.c.id == rid))
    return redirect(url_for("admin_caselaws", mode="delete"))

# --- Contact page (nav safety) ---
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

if __name__ == "__main__":
    # For local dev only; Render uses gunicorn start command
    app.run(debug=True)
