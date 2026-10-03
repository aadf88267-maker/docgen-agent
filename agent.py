"""
DocGenAppWatch Agent - single-file backend.
Natural-language field queries + page fetch + LLM extraction.
Robust against any search response shape.
"""

import os
import json
import requests
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

LOG_FILE = Path("log.json")
RESULTS_FILE = Path("results.json")
USAGE_FILE = Path("search_usage.json")

MAX_SEARCHES_PER_RUN = 160
MONTHLY_CAPS = {"exa": 1200, "firecrawl": 1000}

NTFY_LIVE_TOPIC = "docgen-live-aadf88267"

WORK_LOG = []

def log(kind, **kwargs):
    WORK_LOG.append({"kind": kind, **kwargs})

def live(kind, **kwargs):
    try:
        payload = {"kind": kind, **kwargs}
        requests.post(
            f"https://ntfy.sh/{NTFY_LIVE_TOPIC}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Title": "evt", "Priority": "min"},
            timeout=3,
        )
    except Exception:
        pass

KNOWN_APPS = [
    {"name": "ChatGPT Work", "url": "https://openai.com/chatgpt/pricing/"},
    {"name": "Claude Cowork", "url": "https://www.anthropic.com/pricing"},
    {"name": "Gamma", "url": "https://gamma.app/pricing"},
    {"name": "Notion AI", "url": "https://www.notion.com/pricing"},
    {"name": "Canva Magic Write", "url": "https://www.canva.com/pricing/"},
    {"name": "Adobe Acrobat AI", "url": "https://www.adobe.com/acrobat/pricing.html"},
    {"name": "Microsoft 365 Copilot", "url": "https://www.microsoft.com/microsoft-365/copilot"},
    {"name": "Jotform Document Generator", "url": "https://www.jotform.com/pricing/"},
    {"name": "Slaide", "url": "https://slaide.com/pricing"},
    {"name": "Liner Write", "url": "https://getliner.com/pricing"},
    {"name": "Skywork", "url": "https://skywork.ai/pricing"},
    {"name": "Type", "url": "https://type.ai/pricing"},
    {"name": "Sembly", "url": "https://www.sembly.ai/pricing/"},
    {"name": "QwenWork", "url": "https://qwenwork.aliyun.com/"},
    {"name": "Watto AI", "url": "https://watto.ai"},
    {"name": "Capitol AI", "url": "https://capitol.ai"},
    {"name": "Cobl.ai", "url": "https://cobl.ai"},
    {"name": "Google Gemini Docs", "url": "https://one.google.com/about/plans"},
    {"name": "AI Doc Maker", "url": "https://aidocmaker.com"},
    {"name": "PandaDoc", "url": "https://www.pandadoc.com/pricing/"},
]

FIELD_QUERIES = [
    "what is the free tier of {name}",
    "how much does {name} cost per month",
    "what can you do with {name} features",
    "what file formats does {name} export",
    "is {name} worth it review pros cons",
]

DISCOVERY_QUERIES = [
    "new AI document generator 2026",
    "launch AI document creation tool 2026",
    "Show HN AI document generator",
    "Product Hunt AI document generator 2026",
    "AI document startup funding 2026",
    "YC AI document generator",
    "AI document generation seed round",
    "best new document AI tool 2026",
    "AI report generator launch",
    "AI proposal generator new app",
    "AI contract generator startup",
    "AI slides generator new tool 2026",
    "AI resume generator launch 2026",
    "AI business document generator new",
    "AI doc automation tool 2026",
    "OpenAI document generator update",
    "Anthropic Claude document features",
    "Google Gemini document generation update",
    "Notion AI new features 2026",
    "Gamma AI changelog 2026",
    "Canva Magic Write update",
    "Adobe Acrobat AI new features",
    "Microsoft Copilot document update",
    "AI PDF generator new",
    "AI Word document generator new",
    "AI document summarizer launch",
    "AI document translation tool new",
    "AI legal document generator startup",
    "AI technical writing tool launch",
    "AI invoice generator new app",
]

# ============================================================
# HELPERS — bulletproof against any response shape
# ============================================================

def _extract_hits(data):
    """Find the list of search hits in any common response shape."""
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []

    for key in ("results", "hits", "items", "data", "organic", "web", "pages"):
        v = data.get(key)
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            for sub in ("results", "hits", "items", "data", "organic", "web"):
                sv = v.get(sub)
                if isinstance(sv, list):
                    return sv

    return []

def _safe_snippet(h):
    if not isinstance(h, dict):
        return str(h)[:800] if h else ""
    for key in ("snippet", "description", "text", "content", "markdown", "summary"):
        v = h.get(key)
        if isinstance(v, str) and v:
            return v[:800]
        if isinstance(v, dict):
            for sub in ("text", "value", "content", "snippet"):
                sv = v.get(sub)
                if isinstance(sv, str) and sv:
                    return sv[:800]
        if isinstance(v, list):
            joined = " ".join(str(x) for x in v if isinstance(x, (str, int, float)))
            if joined:
                return joined[:800]
    snips = h.get("snippets") or h.get("highlights")
    if isinstance(snips, list):
        joined = " ".join(str(x) for x in snips if isinstance(x, (str, int, float)))
        if joined:
            return joined[:800]
    return ""

def _safe_url(h):
    if not isinstance(h, dict):
        return ""
    for key in ("url", "link", "href", "source"):
        v = h.get(key)
        if isinstance(v, str):
            return v
    return ""

def _safe_title(h):
    if not isinstance(h, dict):
        return ""
    for key in ("title", "name", "heading"):
        v = h.get(key)
        if isinstance(v, str):
            return v
    return ""

# ============================================================
# SEARCH ROUTER
# ============================================================

def _load_usage():
    if not USAGE_FILE.exists():
        return {}
    try:
        return json.loads(USAGE_FILE.read_text())
    except Exception:
        return {}

def _save_usage(data):
    USAGE_FILE.write_text(json.dumps(data, indent=2))

def _current_month():
    return datetime.now(timezone.utc).strftime("%Y-%m")

def _check_and_increment(provider, cap=None):
    data = _load_usage()
    month = _current_month()
    if data.get("month") != month:
        data = {"month": month, "counts": {}}
    counts = data.setdefault("counts", {})
    used = counts.get(provider, 0)
    if cap is not None and used >= cap:
        return False
    counts[provider] = used + 1
    _save_usage(data)
    return True

def _you_search(query, num=5):
    r = requests.post(
        "https://api.you.com/v1/agents/search",
        headers={"Content-Type": "application/json"},
        json={"query": query, "count": num},
        timeout=20,
    )
    if r.status_code == 402:
        raise RuntimeError("You.com daily limit reached")
    if r.status_code == 429:
        raise RuntimeError("You.com rate limited")
    r.raise_for_status()
    data = r.json()
    hits = _extract_hits(data)
    out = []
    for h in hits[:num]:
        out.append({
            "title": _safe_title(h),
            "url": _safe_url(h),
            "snippet": _safe_snippet(h),
        })
    return out

def _exa_search(query, num=5):
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise RuntimeError("EXA_API_KEY not set")
    r = requests.post(
        "https://api.exa.ai/search",
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json={"query": query, "numResults": num, "contents": {"highlights": True}},
        timeout=25,
    )
    r.raise_for_status()
    data = r.json()
    hits = _extract_hits(data)
    out = []
    for h in hits[:num]:
        out.append({
            "title": _safe_title(h),
            "url": _safe_url(h),
            "snippet": _safe_snippet({"snippets": h.get("highlights", []) or []}),
        })
    return out

def _firecrawl_search(query, num=5):
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise RuntimeError("FIRECRAWL_API_KEY not set")
    r = requests.post(
        "https://api.firecrawl.dev/v1/search",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"query": query, "limit": num},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    hits = _extract_hits(data)
    out = []
    for h in hits[:num]:
        out.append({
            "title": _safe_title(h),
            "url": _safe_url(h),
            "snippet": _safe_snippet(h),
        })
    return out

def search_web(query, num=5):
    try:
        out = _you_search(query, num)
        log("search", query=query, provider="you.com", hits=len(out))
        live("search", query=query, provider="you.com", hits=len(out))
        return out
    except Exception as e:
        log("search", query=query, provider="you.com", error=str(e)[:120])
        live("search", query=query, provider="you.com", error=str(e)[:80])

    if _check_and_increment("exa", MONTHLY_CAPS["exa"]):
        try:
            out = _exa_search(query, num)
            log("search", query=query, provider="exa", hits=len(out))
            live("search", query=query, provider="exa", hits=len(out))
            return out
        except Exception as e:
            log("search", query=query, provider="exa", error=str(e)[:120])
            live("search", query=query, provider="exa", error=str(e)[:80])

    if _check_and_increment("firecrawl", MONTHLY_CAPS["firecrawl"]):
        try:
            out = _firecrawl_search(query, num)
            log("search", query=query, provider="firecrawl", hits=len(out))
            live("search", query=query, provider="firecrawl", hits=len(out))
            return out
        except Exception as e:
            log("search", query=query, provider="firecrawl", error=str(e)[:120])
            live("search", query=query, provider="firecrawl", error=str(e)[:80])

    return []

# ============================================================
# PAGE FETCH
# ============================================================

def fetch_page(url, max_chars=8000):
    if not url:
        return ""
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        return ""
    if not _check_and_increment("firecrawl", MONTHLY_CAPS["firecrawl"]):
        return ""
    try:
        r = requests.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
            timeout=45,
        )
        r.raise_for_status()
        data = r.json()
        inner = data.get("data") if isinstance(data, dict) else None
        md = ""
        if isinstance(inner, dict):
            md = inner.get("markdown") or ""
        log("fetch", url=url, chars=len(md))
        live("fetch", url=url, chars=len(md))
        return md[:max_chars]
    except Exception as e:
        log("fetch", url=url, error=str(e)[:120])
        live("fetch", url=url, error=str(e)[:80])
        return ""

# ============================================================
# LLM ROUTER
# ============================================================

EXTRACTION_PROMPT = """You are a research analyst. Below are (a) targeted search snippets (some from Google AI Overviews) and (b) the full text of the app's pricing page (if available) about a document generation app.

Extract structured data. Prefer the page text over snippets when they conflict. Return ONLY valid JSON matching this exact schema. Use null for unknown — DO NOT guess.

FILL EVERY FIELD YOU CAN. Pay attention to:
- category: general type (document generation, presentations, e-sign, etc.)
- vendor: company that makes the product
- website: product homepage
- free_tier.available: is there a free plan? true/false
- free_tier.credits: exact number of free credits or uses
- free_tier.limits: any restrictions
- paid_pricing.cheapest_plan: cheapest paid tier with price
- paid_pricing.notes: higher tiers, enterprise, per-user pricing notes
- key_features: list of notable features
- output_formats: what it can export
- support.channels: how users get help (email, chat, phone, help_center, community)
- quality_notes: pros/cons sentiment from reviews
- confidence: 0.0-1.0

Schema:
{{
  "app_name": "string",
  "vendor": "string or null",
  "website": "string or null",
  "category": "string",
  "free_tier": {{"available": true/false, "credits": "string or null", "limits": "string or null"}},
  "paid_pricing": {{"cheapest_plan": "string or null", "notes": "string or null"}},
  "key_features": ["string"],
  "output_formats": ["string"],
  "support": {{"channels": ["email","chat","phone","help_center"], "notes": "string or null"}},
  "quality_notes": "string or null",
  "confidence": 0.0
}}

App name: {app_name}

=== SEARCH SNIPPETS ===
{snippets}

=== PRICING PAGE TEXT ===
{page_text}

Return only the JSON object. No markdown. No explanation.
"""

def _client(base_url, api_key):
    return OpenAI(base_url=base_url, api_key=api_key)

def _llm_nemotron(prompt):
    c = _client("https://integrate.api.nvidia.com/v1", os.getenv("NVIDIA_API_KEY"))
    r = c.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=2500,
    )
    return r.choices[0].message.content

def _llm_gemini(prompt):
    c = _client(
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        os.getenv("GEMINI_API_KEY"),
    )
    r = c.chat.completions.create(
        model="gemini-3.5-flash-lite",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=2500,
    )
    return r.choices[0].message.content

def _llm_deepseek_orca(prompt):
    c = _client("https://api.orcarouter.ai/v1", os.getenv("ORCA_API_KEY"))
    r = c.chat.completions.create(
        model="deepseek/deepseek-v4-flash-free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=2500,
    )
    return r.choices[0].message.content

def _llm_kimi_k3(prompt):
    c = _client("https://aihubmix.com/v1", os.getenv("AIHUBMIX_API_KEY"))
    r = c.chat.completions.create(
        model="coding-kimi-k3-free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=2500,
    )
    return r.choices[0].message.content

def _llm_minimax_m3(prompt):
    c = _client("https://aihubmix.com/v1", os.getenv("AIHUBMIX_API_KEY"))
    r = c.chat.completions.create(
        model="coding-minimax-m3-free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=2500,
    )
    return r.choices[0].message.content

def _llm_glm_52(prompt):
    c = _client("https://aihubmix.com/v1", os.getenv("AIHUBMIX_API_KEY"))
    r = c.chat.completions.create(
        model="coding-glm-5.2-free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=2500,
    )
    return r.choices[0].message.content

def _clean_json(raw):
    if raw is None:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(s[start:end + 1])
    except Exception:
        return None

def extract_app(app_name, search_results, page_text=""):
    snippets = "\n\n".join(
        f"- {r['title']} ({r['url']})\n  {r['snippet']}"
        for r in search_results[:25]
    )
    prompt = EXTRACTION_PROMPT.format(
        app_name=app_name,
        snippets=snippets or "(no snippets)",
        page_text=page_text or "(no page text)",
    )

    providers = [
        ("nemotron", _llm_nemotron),
        ("gemini", _llm_gemini),
        ("deepseek-v4-flash-free", _llm_deepseek_orca),
        ("kimi-k3-free", _llm_kimi_k3),
        ("minimax-m3-free", _llm_minimax_m3),
        ("glm-5.2-free", _llm_glm_52),
    ]

    for name, fn in providers:
        try:
            raw = fn(prompt)
            parsed = _clean_json(raw)
            if parsed:
                log("llm", app=app_name, model=name, status="ok")
                live("llm", app=app_name, model=name, status="ok")
                parsed["_extracted_by"] = name
                return parsed
            err = (raw or "")[:80]
            log("llm", app=app_name, model=name, status="invalid_json", error=err)
            live("llm", app=app_name, model=name, status="invalid")
        except Exception as e:
            log("llm", app=app_name, model=name, status="failed", error=str(e)[:200])
            live("llm", app=app_name, model=name, status="failed")

    return {"error": "all_llms_failed", "app_name": app_name}

# ============================================================
# STORAGE
# ============================================================

def load_previous_results():
    if not RESULTS_FILE.exists():
        return {}
    try:
        data = json.loads(RESULTS_FILE.read_text())
        return {item["app_name"]: item for item in data.get("apps", [])}
    except Exception:
        return {}

def save_results(apps, run_meta):
    RESULTS_FILE.write_text(
        json.dumps({"last_run": run_meta, "apps": apps}, indent=2)
    )

def load_log():
    if not LOG_FILE.exists():
        return {"chat": "search x", "pinned": True, "days": []}
    try:
        return json.loads(LOG_FILE.read_text())
    except Exception:
        return {"chat": "search x", "pinned": True, "days": []}

def append_day(day_entry):
    data = load_log()
    days = data.get("days", [])
    days = [d for d in days if d.get("date") != day_entry["date"]]
    days.append(day_entry)
    days.sort(key=lambda d: d.get("date", ""), reverse=True)
    data["days"] = days
    LOG_FILE.write_text(json.dumps(data, indent=2))

def diff_app(old, new):
    if not old:
        return ["new app tracked"]
    changes = []
    for key in ["free_tier", "paid_pricing", "key_features", "support", "output_formats"]:
        if old.get(key) != new.get(key):
            changes.append(f"{key} changed")
    return changes

# ============================================================
# MAIN
# ============================================================

def main():
    global WORK_LOG
    WORK_LOG = []

    started = datetime.now(timezone.utc)
    run_meta = {"started_at": started.isoformat(), "searches_used": 0}
    print(f"[worker] Starting at {run_meta['started_at']}")

    live("start", date=started.strftime("%Y-%m-%d"))

    previous = load_previous_results()
    search_count = 0

    def budget_left():
        return search_count < MAX_SEARCHES_PER_RUN

    new_apps = []
    for app in KNOWN_APPS:
        if not budget_left():
            log("info", message="search budget exhausted")
            break
        name = app["name"]

        results = []
        for tpl in FIELD_QUERIES:
            if not budget_left():
                break
            q = tpl.format(name=name)
            results.extend(search_web(q, num=5))
            search_count += 1

        page_text = fetch_page(app.get("url"), max_chars=8000)

        extracted = extract_app(name, results, page_text=page_text)
        extracted["app_name"] = name
        extracted["source_url"] = app.get("url")
        extracted["_searched_at"] = run_meta["started_at"]
        if not extracted.get("website"):
            extracted["website"] = app.get("url")
        new_apps.append(extracted)

    for q in DISCOVERY_QUERIES:
        if not budget_left():
            break
        search_web(q, num=5)
        search_count += 1

    findings = []
    for app in new_apps:
        changes = diff_app(previous.get(app["app_name"]), app)
        if changes:
            is_new = not previous.get(app["app_name"])
            findings.append({
                "type": "new" if is_new else "change",
                "app": app["app_name"],
                "summary": "new app tracked" if is_new else ", ".join(changes),
                "url": app.get("source_url"),
                "extracted_by": app.get("_extracted_by"),
                "category": app.get("category"),
                "vendor": app.get("vendor"),
                "website": app.get("website"),
                "free_tier": app.get("free_tier"),
                "paid_pricing": app.get("paid_pricing"),
                "key_features": app.get("key_features"),
                "output_formats": app.get("output_formats"),
                "support": app.get("support"),
                "quality_notes": app.get("quality_notes"),
                "confidence": app.get("confidence"),
            })

    run_meta["searches_used"] = search_count
    run_meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    run_meta["apps_tracked"] = len(new_apps)
    run_meta["findings"] = len(findings)

    save_results(new_apps, run_meta)

    day_entry = {
        "date": started.strftime("%Y-%m-%d"),
        "started_at": run_meta["started_at"],
        "finished_at": run_meta["finished_at"],
        "searches": search_count,
        "llm_calls": sum(1 for l in WORK_LOG if l.get("kind") == "llm"),
        "work_log": WORK_LOG,
        "findings": findings,
        "apps": new_apps,
    }
    append_day(day_entry)

    live("end", findings=len(findings), searches=search_count, date=day_entry["date"])

    print(f"[worker] Day appended: {day_entry['date']}, findings: {len(findings)}")
    print(f"[worker] Done. Searches: {search_count}/{MAX_SEARCHES_PER_RUN}")

if __name__ == "__main__":
    main()
