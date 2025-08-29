from flask import Flask, request, render_template, abort, redirect, url_for, flash, send_file, jsonify, render_template_string
from sqlalchemy import create_engine, MetaData, Table, select, or_, func, text, and_, insert, update, delete
from bs4 import BeautifulSoup
import os, re, io, ssl
import pandas as pd
from collections import defaultdict
from datetime import date, datetime

# ======================= DB CONFIG =======================
import os, ssl

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

# TiDB Cloud typically uses port 4000 and requires TLS
_use_tidb_tls = (DB_PORT == "4000") or ("tidbcloud.com" in DB_HOST.lower())

if _use_tidb_tls:
    # On Render, add the CA as a Secret File and set TIDB_SSL_CA to that absolute path.
    ssl_ca = os.getenv("TIDB_SSL_CA")  # e.g., /etc/ssl/certs/isrgrootx1.pem
    if not ssl_ca or not os.path.isfile(ssl_ca):
        raise RuntimeError(
            "TIDB_SSL_CA not set or file not found. On Render, create a Secret File with the CA "
            "and set TIDB_SSL_CA to its absolute path."
        )
    # PyMySQL expects an 'ssl' dict; passing the CA file is enough.
    CONNECT_ARGS = {"ssl": {"ca": ssl_ca}}
    print(f"[DB] TLS enabled with CA: {ssl_ca}")
else:
    print("[DB] TLS not required (non-TiDB or port != 4000).")

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
    "summary_head_note","full_case_law"
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

# ======================= HELPERS =======================
def like_filters(q: str):
    terms = []
    for col in SEARCHABLE_COLS:
        if hasattr(Acer.c, col):
            terms.append(func.lower(getattr(Acer.c, col)).like(f"%{q.lower()}%"))
    return or_(*terms) if terms else text("1=1")

def _distinct(colname):
    with ENGINE.connect() as conn:
        rows = conn.execute(
            select(getattr(Acer.c, colname)).where(
                getattr(Acer.c, colname).isnot(None),
                getattr(Acer.c, colname) != ""
            ).distinct().order_by(getattr(Acer.c, colname).asc())
        ).scalars().all()
    return [r for r in rows if r not in (None, "")]

def _subjects_grouped():
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
    return {k: sorted(v) for k, v in sorted(d.items(), key=lambda kv: kv[0].lower())}

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

# ======================= AI EXPLANATION (existing) =======================
def get_ai_explanation(prompt_text):
    full_prompt = (
        "You are an expert on Indian GST laws. Answer the following query comprehensively in a couple of paragraphs. "
        "Specifically mention relevant sections, rules, notifications, or circulars. "
        "Then, identify keywords or phrases from your answer that could be used to search for related case laws in a database. "
        "List these keywords/phrases (e.g., 'Input Tax Credit', 'Section 18', 'Rule 36') separated by commas.\n\n"
        f"Query: {prompt_text}"
    )
    try:
        return {
            "text": _openai_chat(
                [
                    {"role": "system", "content": "You are a helpful assistant knowledgeable about Indian GST laws."},
                    {"role": "user", "content": full_prompt},
                ],
                max_tokens=1200,
                temperature=0.7,
            )
        }
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
        clauses.append(like_filters(keywords_for_search))
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
        if q:              clauses.append(like_filters(q))
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
    if q:              clauses.append(like_filters(q))
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
        clauses.append(like_filters(ai_q))
    elif q:
        clauses.append(like_filters(q))
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

def _compose_case_lines(rows, max_chars=220):
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
        rows = _fetch_rows_for_current_filters(request.args, cap=60)
        if not rows:
            return jsonify({"ok": False, "message": "No case law results to summarize for the current filters."})

        # Build topic string from chosen filters
        topic = _selected_topic(request.args)
        cases_text = _compose_case_lines(rows, max_chars=220)

        # Focused prompt: ONLY on the chosen filter topic.
        prompt = (
            "You are assisting a GST case-law researcher. Read the list of cases below and produce a concise, paragraph-only synthesis "
            "STRICTLY about the specified topic.\n\n"
            f"Topic: {topic}\n\n"
            "Guidelines:\n"
            "• Base your analysis primarily on the headnotes; only include points relevant to the topic.\n"
            "• Do not drift to unrelated GST themes; ignore material outside the topic.\n"
            "• Cite 3–6 cases by party or case number inline when they support a point.\n"
            "• Write 2–4 tight paragraphs (no bullet points, no numbered lists, no markdown formatting, no asterisks).\n"
            "• Do NOT include generic wrap-ups (e.g., beginning with 'Overall', 'In conclusion', 'In summary').\n\n"
            f"Cases:\n{cases_text}\n\n"
            "Now write the synthesis:"
        )

        raw = _openai_chat(
            [
                {"role": "system", "content": "You are a concise legal analyst for Indian GST jurisprudence."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=700, temperature=0.4
        )

        summary = _clean_summary(raw)
        if not summary:
            return jsonify({"ok": False, "message": "The model returned an empty summary. Try refining filters and retry."})
        return jsonify({"ok": True, "summary": summary})

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
    where_clause = like_filters(q) if q else text("1=1")
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
    app.run(debug=True)
