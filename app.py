from flask import Flask, request, render_template, abort, redirect, url_for, flash, send_file, jsonify, render_template_string
from sqlalchemy import create_engine, MetaData, Table, select, or_, func, text, and_, insert, update, delete
from bs4 import BeautifulSoup
from werkzeug.utils import secure_filename
import os, re, io, ssl, json, time, hashlib
import pandas as pd
from collections import defaultdict
from datetime import date, datetime

# ======================= FLASK & UPLOAD CONFIG =======================
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB upload cap
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

# ======================= SEARCH HELPERS =======================
def like_filters(q: str):
    """
    OR across SEARCHABLE_COLS using LIKE (case-insensitive under typical MySQL *_ci collations).
    NOTE: no LOWER() wrapper so indexes/collation can help -> faster.
    """
    terms = []
    for col in SEARCHABLE_COLS:
        if hasattr(Acer.c, col):
            terms.append(getattr(Acer.c, col).like(f"%{q}%"))
    return or_(*terms) if terms else text("1=1")

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
    if not q: return []
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
            if tok in _STOPWORDS: continue
            if len(tok) <= 2 and tok not in _KEEP_SHORT: continue
            if tok not in seen:
                kept.append(tok); seen.add(tok)
    return kept

def _nlq_clause(query: str):
    toks = _nlq_keywords(query)
    if not toks:
        return like_filters(query)
    # AND across tokens; each token is OR across columns
    return and_(*[like_filters(t) for t in toks])

def _merge_ai_and_user_keywords(ai_text: str, user_query: str) -> str:
    ai_tokens = _nlq_keywords(ai_text or "")
    ai_compact = " ".join(ai_tokens) if ai_tokens else ""
    user_query = (user_query or "").strip()
    if ai_compact and user_query:
        return f"{ai_compact} {user_query}"
    return ai_compact or user_query

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
        ln = re.sub(r'^\s*[\*\-•]+\s*', '', ln)
        ln = re.sub(r'^\s*\d+\.\s*', '', ln)
        lines.append(ln)
    text = "\n".join(lines)
    text = re.sub(r'(?im)^\s*(overall|in summary|in conclusion|to conclude|in short)\b.*$', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text

# ======================= AI EXPLANATION (DETAILED + NO CASE NAMES) =======================
def get_ai_explanation(prompt_text):
    args = request.args
    key_src = "|".join([
        "ai_explain_v3", (prompt_text or "").strip(),
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

    context_bits = []
    if args.getlist("section"): context_bits.append("Sections: " + ", ".join(args.getlist("section")))
    if args.getlist("rule"):    context_bits.append("Rules: " + ", ".join(args.getlist("rule")))
    if args.getlist("industry"):context_bits.append("Industry: " + ", ".join(args.getlist("industry")))
    if args.getlist("s1"):      context_bits.append("Subject-1: " + ", ".join(args.getlist("s1")))
    if args.getlist("s2"):      context_bits.append("Subject-2: " + ", ".join(args.getlist("s2")))
    filter_context = (" | ".join(context_bits)) if context_bits else "General GST"

    full_prompt = (
        "You are an expert on Indian GST. Write 2–4 compact paragraphs that explain ONLY the statutory position: "
        "focus on CGST/IGST Acts, Rules, relevant Notifications and Circulars. "
        "Do NOT mention or invent ANY case names, parties, judges, or case citations. "
        "NO bullet points, NO generic wrap-ups; just crisp paragraphs aligned to the query and filters.\n\n"
        f"Filter Context: {filter_context}\n"
        f"User Query: {(prompt_text or '').strip()}\n"
    )
    try:
        text = _openai_chat(
            [
                {"role": "system", "content": "You are a precise Indian GST legal analyst. Never cite or name cases; stick to sections, rules, notifications, circulars."},
                {"role": "user", "content": full_prompt},
            ],
            max_tokens=700, temperature=0.3,
        )
        _ai_cache_set(cache_key, text, ttl_sec=900)
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
    q_raw          = (request.args.get("q") or "").strip()

    results, total, pages = [], 0, 1
    ai_response_data = None

    COMMON_COLS = [
        Acer.c.id, Acer.c.case_law_number, Acer.c.name_of_party, Acer.c.date,
        Acer.c.summary_head_note,
        Acer.c.type_of_court, Acer.c.state, Acer.c.year,
        Acer.c.section, Acer.c.rule, Acer.c.citation,
        Acer.c.subject_matter1, Acer.c.subject_matter2,
    ]

    page = max(int(request.args.get("page", 1) or 1), 1)
    per_page = 10
    offset = (page - 1) * per_page

    def _run_expr(expr):
        with ENGINE.connect() as conn:
            _total = conn.execute(select(func.count()).select_from(Acer).where(expr)).scalar_one()
            _rows = conn.execute(
                select(*COMMON_COLS).where(expr).order_by(Acer.c.id.desc()).limit(per_page).offset(offset)
            ).mappings().all()
        return _total, [dict(r) for r in _rows]

    # ---------- AI search ----------
    if is_ai_search and ai_q:
        ai_response_data = get_ai_explanation(ai_q)
        ai_text = ai_response_data.get('text', '') if isinstance(ai_response_data, dict) else ''
        merged_search_text = _merge_ai_and_user_keywords(ai_text, ai_q)

        base_filters = []
        if sel_industries: base_filters.append(Acer.c.industry_sector.in_(sel_industries))
        if sel_sections:   base_filters.append(Acer.c.section.in_(sel_sections))
        if sel_rules:      base_filters.append(Acer.c.rule.in_(sel_rules))
        if sel_parties:    base_filters.append(Acer.c.name_of_party.in_(sel_parties))
        if sel_s1:         base_filters.append(Acer.c.subject_matter1.in_(sel_s1))
        if sel_s2:         base_filters.append(Acer.c.subject_matter2.in_(sel_s2))

        # Try 1: NLQ over (AI tokens + user query)
        where_expr = and_(*(base_filters + [ _nlq_clause(merged_search_text) ])) if base_filters else _nlq_clause(merged_search_text)
        total, results = _run_expr(where_expr)

        # Try 2: NLQ over raw user query
        if total == 0:
            where_expr = and_(*(base_filters + [ _nlq_clause(ai_q) ])) if base_filters else _nlq_clause(ai_q)
            total, results = _run_expr(where_expr)

        # Try 3: Simple LIKE over raw user query
        if total == 0:
            where_expr = and_(*(base_filters + [ like_filters(ai_q) ])) if base_filters else like_filters(ai_q)
            total, results = _run_expr(where_expr)

        pages = (total // per_page) + (1 if total % per_page else 0)
        q_display = ai_q  # ensures results block renders even in AI mode

    # ---------- Standard search (with fallback NLQ -> LIKE) ----------
    elif q_raw or any([sel_industries, sel_sections, sel_rules, sel_parties, sel_s1, sel_s2]):
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

    # Pagination links
    def page_url(n):
        params = {
            'industry': sel_industries, 'section': sel_sections, 'rule': sel_rules,
            'party': sel_parties, 's1': sel_s1, 's2': sel_s2, 'page': n
        }
        if is_ai_search:
            params['ai_search'] = 'true'; params['ai_q'] = ai_q
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
        ai_response=ai_response_data, ai_q=ai_q
    )

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
@app.get("/admin")
def admin_root():
    return redirect(url_for("admin_caselaws"))

@app.get("/admin/caselaws")
def admin_caselaws():
    q = (request.args.get("q") or "").strip()
    page = max(int(request.args.get("page", 1)), 1)
    per_page = min(max(int(request.args.get("per_page", 20)), 1), 200)
    offset = (page - 1) * per_page
    where_clause = _nlq_clause(q) if q else text("1=1")
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

# ---- EXPORT CSV (primary) ----
@app.get("/admin/export_csv")
def admin_export_csv():
    cols = [getattr(Acer.c, c) for c in ALL_COLS if hasattr(Acer.c, c)]
    with ENGINE.connect() as conn:
        rows = conn.execute(select(*cols).order_by(Acer.c.id.desc())).mappings().all()
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[c.name for c in cols])
    buf = io.BytesIO()
    # Excel-friendly: UTF-8 with BOM
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    buf.seek(0)
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="cases.csv")

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
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "year" in df.columns:
        def _to_year(v):
            if pd.isna(v): return None
            try:
                s = str(v).strip()
                if len(s) >= 4 and s[:4].isdigit(): return int(s[:4])
                return int(float(s))
            except Exception:
                return None
        df["year"] = df["year"].map(_to_year)
        if "date" in df.columns:
            df["year"] = df["year"].fillna(df["date"].dt.year)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda x: (str(x).strip() if x is not None else None))
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

@app.route("/admin/bulk_upload", methods=["GET","POST"])
def admin_bulk_upload():
    if request.method == "GET":
        try:
            return render_template("admin_bulk_upload.html")
        except Exception:
            html = """
            <h2>Bulk Upload Case Laws (Excel/CSV)</h2>
            <p>Accepted: .xlsx, .xls, .csv</p>
            <p>
              <a href="{{ url_for('admin_bulk_template') }}">Download Upload Template</a> |
              <a href="{{ url_for('admin_bulk_download') }}">Bulk Download All Cases</a>
            </p>
            <form method="post" enctype="multipart/form-data">
              <input type="file" name="file" required>
              <button type="submit">Upload</button>
              <a href="{{ url_for('admin_caselaws') }}">Back to Admin</a>
            </form>
            """
            return render_template_string(html)

    f = request.files.get("file")
    if not f or not getattr(f, "filename", ""):
        flash("No file selected.", "err")
        return redirect(url_for("admin_bulk_upload"))
    filename = secure_filename(f.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_UPLOADS:
        flash("Unsupported file type. Please upload .xlsx, .xls or .csv", "err")
        return redirect(url_for("admin_bulk_upload"))

    try:
        data = f.read()
        if ext in {".xlsx",".xls"}:
            df = pd.read_excel(io.BytesIO(data))
        else:
            df = pd.read_csv(io.BytesIO(data))
    except Exception as e:
        flash(f"Failed to read file: {e}", "err")
        return redirect(url_for("admin_bulk_upload"))

    if df.empty:
        flash("Uploaded file is empty.", "err")
        return redirect(url_for("admin_bulk_upload"))

    df = _normalize_columns(df)
    if df.empty:
        flash("No recognizable columns found. Use the template headers.", "err")
        return redirect(url_for("admin_bulk_upload"))

    df = _coerce_types(df)
    if len(df) > 5000:
        df = df.iloc[:5000, :]

    records = []
    for _, row in df.iterrows():
        rec = {}
        for c in _TEMPLATE_COLUMNS:
            if c in df.columns:
                v = row[c]
                if pd.isna(v): v = None
                rec[c] = v
        if any(v not in (None, "") for v in rec.values()):
            records.append(rec)

    if not records:
        flash("No valid rows to insert.", "err")
        return redirect(url_for("admin_bulk_upload"))

    try:
        with ENGINE.begin() as conn:
            conn.execute(insert(Acer), records)
    except Exception as e:
        flash(f"Insert failed: {e}", "err")
        return redirect(url_for("admin_bulk_upload"))

    flash(f"Uploaded and inserted {len(records)} row(s).", "ok")
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

if __name__ == "__main__":
    app.run(debug=True)
