"""
DocGenAppWatch Agent — dual-mode backend.

AGENT_MODE=watch    → monitor known apps + find new competitors
AGENT_MODE=research → scan ANY sector with user-defined fields
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
BASELINE_FILE = Path("baseline.json")
USAGE_FILE = Path("search_usage.json")

MODE = os.getenv("AGENT_MODE", "watch").lower()
RESEARCH_SECTOR = os.getenv("RESEARCH_SECTOR", "").strip()[:200] or "AI document generation tools"
RESEARCH_FIELDS_RAW = os.getenv("RESEARCH_FIELDS", "").strip()[:500]

def _parse_fields(raw):
    if not raw:
        return ["description", "url", "category"]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts[:12]

RESEARCH_FIELDS = _parse_fields(RESEARCH_FIELDS_RAW)

MAX_SEARCHES_PER_RUN = 120
DAILY_EXA_CAP = 40
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

# ============================================================
# KNOWN APPS (WATCH only)
# ============================================================

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

WATCH_QUERIES = [
    "{name} new update announcement 2026",
    "{name} pricing or feature change 2026",
]

COMPETITOR_QUERIES = [
    "new AI document generator launched this week",
    "new AI writing assistant launch 2026",
    "Product Hunt document tool this month",
    "AI presentation tool new release",
    "AI contract generator new launch",
    "YC batch AI document startup",
    "AI proposal software new tool 2026",
    "just launched AI doc app",
    "new entrant document automation",
    "AI document startup funding this week",
    "new AI spreadsheet tool 2026",
    "AI report generator launch this month",
    "Notion alternative new 2026",
    "Gamma alternative new tool",
    "AI doc app Show HN 2026",
    "AI powerpoint generator new",
    "AI resume tool new launch",
    "AI invoice generator new 2026",
    "AI legal document startup 2026",
    "AI technical doc generator new",
]

# ============================================================
# HELPERS
# ============================================================

def _extract_hits(data):
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

def _sector_queries(sector):
    s = sector.strip()
    return [
        f"{s} list 2026",
        f"best {s} 2026",
        f"top {s} 2026",
        f"{s} comparison 2026",
        f"{s} roundup 2026",
        f"new {s} 2026",
        f"{s} guide 2026",
        f"{s} database 2026",
        f"{s} directory",
        f"popular {s}",
        f"{s} examples",
        f"{s} list with details",
        f"{s} review site",
        f"{s} ranked 2026",
        f"most used {s}",
        f"{s} overview 2026",
        f"{s} wiki",
        f"{s} facts",
        f"{s} data 2026",
        f"{s} stats 2026",
    ]

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

def _today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def _check_and_increment(provider, cap=None):
    data = _load_usage()
    month = _current_month()
    day = _today()
    if data.get("month") != month:
        data = {"month": month, "counts": {}, "daily": {}}
    if data.get("daily_date") != day:
        data["daily_date"] = day
        data["daily"] = {}
    counts = data.setdefault("counts", {})
    daily = data.setdefault("daily", {})
    used_total = counts.get(provider, 0)
    used_today = daily.get(provider, 0)
    if cap is not None and used_total >= cap:
        return False
    counts[provider] = used_total + 1
    daily[provider] = used_today + 1
    _save_usage(data)
    return True

def _exa_daily_ok():
    data = _load_usage()
    day = _today()
    if data.get("daily_date") != day:
        return True
    return data.get("daily", {}).get("exa", 0) < DAILY_EXA_CAP

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
    return [{"title": _safe_title(h), "url": _safe_url(h), "snippet": _safe_snippet(h)} for h in hits[:num]]

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
    return [
        {
            "title": _safe_title(h),
            "url": _safe_url(h),
            "snippet": _safe_snippet({"snippets": h.get("highlights", []) or []}),
        }
        for h in hits[:num]
    ]

def _firecrawl_search(query, num=5):
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise RuntimeError("FIRECRAWL_API_KEY not set")
    r = requests.post(
        "https://api.firecrawl.dev/v1/search",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"query": query, "limit": num},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    hits = _extract_hits(data)
    return [{"title": _safe_title(h), "url": _safe_url(h), "snippet": _safe_snippet(h)} for h in hits[:num]]

def search_web(query, num=5):
    try:
        out = _you_search(query, num)
        log("search", query=query, provider="you.com", hits=len(out))
        live("search", query=query, provider="you.com", hits=len(out))
        return out
    except Exception as e:
        log("search", query=query, provider="you.com", error=str(e)[:120])
        live("search", query=query, provider="you.com", error=str(e)[:80])

    if _exa_daily_ok() and _check_and_increment("exa", MONTHLY_CAPS["exa"]):
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
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
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

def _client(base_url, api_key):
    return OpenAI(base_url=base_url, api_key=api_key)

def _llm_nemotron(prompt):
    c = _client("https://integrate.api.nvidia.com/v1", os.getenv("NVIDIA_API_KEY"))
    r = c.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=3000,
    )
    return r.choices[0].message.content

def _llm_gemini(prompt):
    c = _client("https://generativelanguage.googleapis.com/v1beta/openai/", os.getenv("GEMINI_API_KEY"))
    r = c.chat.completions.create(
        model="gemini-3.5-flash-lite",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=3000,
    )
    return r.choices[0].message.content

def _llm_deepseek(prompt):
    c = _client("https://api.orcarouter.ai/v1", os.getenv("ORCA_API_KEY"))
    r = c.chat.completions.create(
        model="deepseek/deepseek-v4-flash-free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=3000,
    )
    return r.choices[0].message.content

def _llm_kimi(prompt):
    c = _client("https://aihubmix.com/v1", os.getenv("AIHUBMIX_API_KEY"))
    r = c.chat.completions.create(
        model="coding-kimi-k3-free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=3000,
    )
    return r.choices[0].message.content

def _llm_minimax(prompt):
    c = _client("https://aihubmix.com/v1", os.getenv("AIHUBMIX_API_KEY"))
    r = c.chat.completions.create(
        model="coding-minimax-m3-free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=3000,
    )
    return r.choices[0].message.content

def _llm_glm(prompt):
    c = _client("https://aihubmix.com/v1", os.getenv("AIHUBMIX_API_KEY"))
    r = c.chat.completions.create(
        model="coding-glm-5.2-free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=3000,
    )
    return r.choices[0].message.content

PROVIDERS = [
    ("nemotron", _llm_nemotron),
    ("gemini", _llm_gemini),
    ("deepseek", _llm_deepseek),
    ("kimi", _llm_kimi),
    ("minimax", _llm_minimax),
    ("glm", _llm_glm),
]

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

def _call_llm(prompt, tag):
    for name, fn in PROVIDERS:
        try:
            raw = fn(prompt)
            parsed = _clean_json(raw)
            if parsed:
                log("llm", task=tag, model=name, status="ok")
                live("llm", task=tag, model=name, status="ok")
                parsed["_extracted_by"] = name
                return parsed
            err = (raw or "")[:80]
            log("llm", task=tag, model=name, status="invalid_json", error=err)
        except Exception as e:
            log("llm", task=tag, model=name, status="failed", error=str(e)[:200])
            live("llm", task=tag, model=name, status="failed")
    return None

def _call_llm_text(prompt, tag):
    for name, fn in PROVIDERS:
        try:
            raw = fn(prompt)
            if raw and raw.strip():
                log("llm", task=tag, model=name, status="ok")
                live("llm", task=tag, model=name, status="ok")
                return raw.strip(), name
            log("llm", task=tag, model=name, status="empty")
        except Exception as e:
            log("llm", task=tag, model=name, status="failed", error=str(e)[:200])
            live("llm", task=tag, model=name, status="failed")
    return None, None

# ============================================================
# PROMPTS
# ============================================================

EXTRACT_PROMPT = """You are a research analyst. Below are (a) search snippets and (b) the app's pricing page text.

Extract structured data. Prefer page text over snippets when they conflict. Return ONLY valid JSON matching this schema. Use null for unknown.

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

=== SNIPPETS ===
{snippets}

=== PAGE TEXT ===
{page_text}

Return only the JSON object.
"""

DIFF_PROMPT = """You compare two snapshots of a software product to find meaningful changes.

BASELINE (previous state):
{baseline}

CURRENT (today's state):
{current}

Output 0-3 short change lines, one per line. Examples:
Pro plan: $20 -> $25
New feature: AI Agents
Removed: Zapier integration
Free tier: 400 -> 500 credits

Rules:
- Only report PRICING, FEATURES, SUPPORT, or FREE TIER changes
- If nothing meaningful changed, output exactly: NO_CHANGES
- No bullets, no numbering, no markdown. Plain lines separated by newlines.
- Maximum 3 lines.
"""

COMPETITOR_PROMPT = """Below are web search results about NEW AI document generation tools.

Extract a list of AI document tools that appear NEW (launched or funded in 2025-2026). Skip established players (Notion, Canva, Adobe, Microsoft, OpenAI, Anthropic, Google, Gamma).

Return ONLY valid JSON:
{{
  "apps": [
    {{
      "name": "string",
      "url": "string or null",
      "category": "string",
      "one_liner": "short description",
      "evidence": "why new"
    }}
  ]
}}

Search snippets:
{content}

Return only JSON. No markdown.
"""

def build_research_prompt(sector, fields):
    field_lines = "\n".join(f'- "{f}"' for f in fields)
    field_keys = ", ".join(f'"{f}"' for f in fields)
    example = ",\n      ".join(f'"{f}": "value or null"' for f in fields)
    return f"""You are scanning the market for this sector: {sector}

Below are web search results. Extract EVERY distinct item/product/entry you can find that fits the sector "{sector}".

For each item, extract these specific fields:
{field_lines}

For each field:
- Use a concise string value (number + unit if applicable)
- Use null if the info isn't in the snippets
- DO NOT guess or hallucinate values

Return ONLY valid JSON:
{{
  "items": [
    {{
      "name": "string — name of the item",
      {example}
    }}
  ]
}}

Search snippets:
{{content}}

Return only JSON. No markdown. Maximum 40 items.
"""

REPORT_PROMPT = """You are a market analyst. You just scanned the "{sector}" sector.

Here are the items you found:

{items}

Write a SHORT structured report (200-400 words). Use this exact format:

## Overview
1-2 sentences on what this sector looks like right now.

## Key Players
Bullet list of the top 5-8 most notable items.

## Trends
2-3 bullets on what's happening: pricing patterns, new launches, features, funding, consolidation.

## Gaps & Opportunities
1-3 bullets on what appears missing or underserved.

## Bottom Line
One sentence: what should someone entering this space know?

Rules:
- Return PLAIN TEXT with markdown-style headers (##)
- No JSON
- Be specific — reference actual item names
- If data is sparse, say so honestly
"""

# ============================================================
# STORAGE
# ============================================================

def load_log():
    if not LOG_FILE.exists():
        return {"chat": "search x", "pinned": True, "days": [], "research": []}
    try:
        d = json.loads(LOG_FILE.read_text())
        d.setdefault("days", [])
        d.setdefault("research", [])
        return d
    except Exception:
        return {"chat": "search x", "pinned": True, "days": [], "research": []}

def save_log(data):
    LOG_FILE.write_text(json.dumps(data, indent=2))

def append_day(entry):
    data = load_log()
    days = [d for d in data.get("days", []) if d.get("date") != entry["date"]]
    days.append(entry)
    days.sort(key=lambda d: d.get("date", ""), reverse=True)
    data["days"] = days
    save_log(data)

def append_research(entry):
    data = load_log()
    research = data.get("research", [])
    # If same date+sector exists, replace it
    research = [
        r for r in research
        if not (r.get("date") == entry["date"] and r.get("sector") == entry.get("sector"))
    ]
    research.append(entry)
    research = research[-60:]
    data["research"] = research
    save_log(data)

def load_baseline():
    if not BASELINE_FILE.exists():
        return {"apps": {}}
    try:
        return json.loads(BASELINE_FILE.read_text())
    except Exception:
        return {"apps": {}}

def save_baseline(data):
    BASELINE_FILE.write_text(json.dumps(data, indent=2))

# ============================================================
# WATCH MODE
# ============================================================

def run_watch():
    started = datetime.now(timezone.utc)
    live("start", date=started.strftime("%Y-%m-%d"), mode="watch")
    print(f"[watch] Starting at {started.isoformat()}")

    baseline = load_baseline()
    baseline_apps = baseline.get("apps", {})
    search_count = 0
    updates = []
    checked_apps = []

    def budget_left():
        return search_count < MAX_SEARCHES_PER_RUN

    for app in KNOWN_APPS:
        if not budget_left():
            break
        name = app["name"]
        results = []
        for tpl in WATCH_QUERIES:
            if not budget_left():
                break
            results.extend(search_web(tpl.format(name=name), num=5))
            search_count += 1
        page_text = fetch_page(app.get("url"), max_chars=6000)
        extracted = _call_llm(
            EXTRACT_PROMPT.format(
                app_name=name,
                snippets="\n\n".join(f"- {r['title']} ({r['url']})\n  {r['snippet']}" for r in results[:20]) or "(none)",
                page_text=page_text or "(none)",
            ),
            tag=f"extract:{name}",
        )
        if not extracted:
            extracted = {"error": "all_llms_failed", "app_name": name}
        extracted["app_name"] = name
        extracted["source_url"] = app.get("url")
        if not extracted.get("website"):
            extracted["website"] = app.get("url")
        checked_apps.append(extracted)

        prev = baseline_apps.get(name)
        if prev and not prev.get("error"):
            diff_raw, _ = _call_llm_text(
                DIFF_PROMPT.format(
                    baseline=json.dumps(prev, indent=2)[:3500],
                    current=json.dumps(extracted, indent=2)[:3500],
                ),
                tag=f"diff:{name}",
            )
            if diff_raw and diff_raw.strip() != "NO_CHANGES":
                lines = [l.strip(" -•*") for l in diff_raw.splitlines() if l.strip()]
                for ln in lines[:3]:
                    if ln.upper() == "NO_CHANGES":
                        continue
                    updates.append({"app": name, "change": ln})
        else:
            updates.append({"app": name, "change": "tracking started"})

    new_competitors = []
    all_snippets = []
    for q in COMPETITOR_QUERIES:
        if not budget_left():
            break
        results = search_web(q, num=5)
        search_count += 1
        all_snippets.extend(results[:3])

    if all_snippets:
        comp_raw = _call_llm(
            COMPETITOR_PROMPT.format(
                content="\n\n".join(f"- {r['title']} ({r['url']})\n  {r['snippet']}" for r in all_snippets[:60])
            ),
            tag="competitor_scan",
        )
        if comp_raw and comp_raw.get("apps"):
            for a in comp_raw["apps"][:30]:
                new_competitors.append({
                    "name": a.get("name", ""),
                    "url": a.get("url"),
                    "category": a.get("category"),
                    "one_liner": a.get("one_liner"),
                    "evidence": a.get("evidence"),
                })

    for app in checked_apps:
        if not app.get("error"):
            baseline_apps[app["app_name"]] = app
    baseline["apps"] = baseline_apps
    baseline["updated_at"] = started.isoformat()
    save_baseline(baseline)

    day_entry = {
        "date": started.strftime("%Y-%m-%d"),
        "mode": "watch",
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "searches": search_count,
        "updates": updates,
        "new_competitors": new_competitors,
        "apps_checked": checked_apps,
        "work_log": WORK_LOG,
    }
    append_day(day_entry)
    live("end", date=day_entry["date"], mode="watch",
         updates=len(updates), competitors=len(new_competitors))
    print(f"[watch] Done. Updates: {len(updates)} · Competitors: {len(new_competitors)} · Searches: {search_count}")

# ============================================================
# RESEARCH MODE (dynamic sector + fields + report)
# ============================================================

def run_research():
    started = datetime.now(timezone.utc)
    sector = RESEARCH_SECTOR
    fields = RESEARCH_FIELDS

    live("start", date=started.strftime("%Y-%m-%d"), mode="research", sector=sector, fields=fields)
    print(f"[research] Sector: {sector}")
    print(f"[research] Fields: {fields}")
    print(f"[research] Starting at {started.isoformat()}")

    search_count = 0
    all_results = []
    queries = _sector_queries(sector)

    for q in queries:
        if search_count >= MAX_SEARCHES_PER_RUN:
            break
        results = search_web(q, num=5)
        search_count += 1
        all_results.extend(results)

    content = "\n\n".join(
        f"- {r['title']} ({r['url']})\n  {r['snippet']}"
        for r in all_results[:100]
    )

    prompt = build_research_prompt(sector, fields).format(content=content[:16000] or "(none)")
    extracted = _call_llm(prompt, tag=f"market_scan:{sector}")

    found = []
    if extracted and extracted.get("items"):
        for a in extracted["items"][:60]:
            item = {"name": a.get("name", "")}
            for f in fields:
                item[f] = a.get(f)
            found.append(item)

    # ---- Generate structured report ----
    report = ""
    if found:
        items_txt = json.dumps(found[:40], indent=2)[:8000]
        report, _ = _call_llm_text(
            REPORT_PROMPT.format(sector=sector, items=items_txt),
            tag=f"report:{sector}",
        )
        report = report or ""

    entry = {
        "date": started.strftime("%Y-%m-%d"),
        "mode": "research",
        "sector": sector,
        "fields": fields,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "searches": search_count,
        "found": found,
        "report": report,
        "work_log": WORK_LOG,
    }
    append_research(entry)
    live("end", date=entry["date"], mode="research", sector=sector, found=len(found))
    print(f"[research] Done. Found: {len(found)} items · Report: {len(report)} chars · Searches: {search_count}")

# ============================================================
# MAIN
# ============================================================

def main():
    global WORK_LOG
    WORK_LOG = []
    print(f"[worker] Mode: {MODE}")
    if MODE == "research":
        run_research()
    else:
        run_watch()

if __name__ == "__main__":
    main()
