"""
DocGenAppWatch Agent - single-file backend.

Search chain:  You.com (keyless) -> Exa -> Firecrawl
LLM chain:     Nemotron 3 Super -> Gemini 3.5 Flash-Lite -> DeepSeek V4 Flash (Orca) -> Kimi K3 (AIHubMix) -> MiniMax M3 (AIHubMix) -> GLM 5.2 (AIHubMix)
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

RESULTS_FILE = Path("results.json")
HISTORY_FILE = Path("history.json")
USAGE_FILE = Path("search_usage.json")

MAX_SEARCHES_PER_RUN = 140
MONTHLY_CAPS = {"exa": 1200, "firecrawl": 1000}

NTFY_TOPIC = os.getenv("NTFY_TOPIC", "")

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
    hits = data.get("results") or data.get("hits") or []
    out = []
    for h in hits[:num]:
        out.append({
            "title": h.get("title", ""),
            "url": h.get("url", ""),
            "snippet": (
                h.get("snippet")
                or h.get("description")
                or " ".join(h.get("snippets", []) or [])
            )[:800],
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
    return [
        {
            "title": h.get("title", ""),
            "url": h.get("url", ""),
            "snippet": " ".join(h.get("highlights", []) or [])[:800],
        }
        for h in data.get("results", [])
    ]

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
    items = data.get("data", [])
    return [
        {
            "title": h.get("title", ""),
            "url": h.get("url", ""),
            "snippet": (h.get("description") or h.get("markdown") or "")[:800],
        }
        for h in items
    ]

def search_web(query, num=5):
    try:
        return _you_search(query, num)
    except Exception as e:
        print(f"[search] You.com failed: {e}")

    if _check_and_increment("exa", MONTHLY_CAPS["exa"]):
        try:
            return _exa_search(query, num)
        except Exception as e:
            print(f"[search] Exa failed: {e}")

    if _check_and_increment("firecrawl", MONTHLY_CAPS["firecrawl"]):
        try:
            return _firecrawl_search(query, num)
        except Exception as e:
            print(f"[search] Firecrawl failed: {e}")

    return []

# ============================================================
# LLM ROUTER
# ============================================================

EXTRACTION_PROMPT = """You are a research analyst. Below are web search results about a document generation app.

Extract structured data. Return ONLY valid JSON matching this exact schema. Use null for unknown.

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

Search results:
{content}

Return only the JSON object. No markdown. No explanation.
"""

def _client(base_url, api_key):
    return OpenAI(base_url=base_url, api_key=api_key)

# --- 1. NVIDIA Nemotron 3 Super 120B ---
def _llm_nemotron(prompt):
    c = _client("https://integrate.api.nvidia.com/v1", os.getenv("NVIDIA_API_KEY"))
    r = c.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2000,
    )
    return r.choices[0].message.content

# --- 2. Gemini 3.5 Flash-Lite ---
def _llm_gemini(prompt):
    c = _client(
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        os.getenv("GEMINI_API_KEY"),
    )
    r = c.chat.completions.create(
        model="gemini-3.5-flash-lite",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2000,
    )
    return r.choices[0].message.content

# --- 3. DeepSeek V4 Flash Free via Orca Router ---
def _llm_deepseek_orca(prompt):
    c = _client("https://api.orcarouter.ai/v1", os.getenv("ORCA_API_KEY"))
    r = c.chat.completions.create(
        model="deepseek/deepseek-v4-flash-free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2000,
    )
    return r.choices[0].message.content

# --- 4. Kimi K3 Free via AIHubMix ---
def _llm_kimi_k3(prompt):
    c = _client("https://aihubmix.com/v1", os.getenv("AIHUBMIX_API_KEY"))
    r = c.chat.completions.create(
        model="coding-kimi-k3-free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2000,
    )
    return r.choices[0].message.content

# --- 5. MiniMax M3 Free via AIHubMix ---
def _llm_minimax_m3(prompt):
    c = _client("https://aihubmix.com/v1", os.getenv("AIHUBMIX_API_KEY"))
    r = c.chat.completions.create(
        model="coding-minimax-m3-free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2000,
    )
    return r.choices[0].message.content

# --- 6. GLM 5.2 Free via AIHubMix ---
def _llm_glm_52(prompt):
    c = _client("https://aihubmix.com/v1", os.getenv("AIHUBMIX_API_KEY"))
    r = c.chat.completions.create(
        model="coding-glm-5.2-free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2000,
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

def extract_app(app_name, search_results):
    content = "\n\n".join(
        f"- {r['title']} ({r['url']})\n  {r['snippet']}"
        for r in search_results[:10]
    )
    prompt = EXTRACTION_PROMPT.format(app_name=app_name, content=content)

    providers = [
        ("nemotron", _llm_nemotron),
        ("gemini-3.5-flash-lite", _llm_gemini),
        ("deepseek-v4-flash-free", _llm_deepseek_orca),
        ("kimi-k3-free", _llm_kimi_k3),
        ("minimax-m3-free", _llm_minimax_m3),
        ("glm-5.2-free", _llm_glm_52),
    ]

    for name, fn in providers:
        try:
            parsed = _clean_json(fn(prompt))
            if parsed:
                parsed["_extracted_by"] = name
                return parsed
            print(f"[llm] {name} returned invalid JSON")
        except Exception as e:
            print(f"[llm] {name} failed: {e}")

    return {"error": "all_llms_failed", "app_name": app_name}

# ============================================================
# DIFF + NOTIFY + STORAGE
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

def append_history(changes, run_meta):
    history = []
    if HISTORY_FILE.exists():
        try:
            history = json.loads(HISTORY_FILE.read_text())
        except Exception:
            history = []
    history.append({"run": run_meta, "changes": changes})
    HISTORY_FILE.write_text(json.dumps(history[-90:], indent=2))

def notify_ntfy(title, message):
    if not NTFY_TOPIC:
        print("[notify] NTFY_TOPIC not set, skipping")
        return
    try:
        requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Tags": "robot"},
            timeout=10,
        )
    except Exception as e:
        print(f"[notify] failed: {e}")

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
    started = datetime.now(timezone.utc)
    run_meta = {"started_at": started.isoformat(), "searches_used": 0}
    print(f"[worker] Starting at {run_meta['started_at']}")

    previous = load_previous_results()
    search_count = 0

    def budget_left():
        return search_count < MAX_SEARCHES_PER_RUN

    new_apps = []

    for app in KNOWN_APPS:
        if not budget_left():
            print("[worker] Search budget exhausted")
            break

        name = app["name"]
        results = []
        for q in [
            f"{name} pricing 2026",
            f"{name} new features 2026",
            f"{name} free plan changelog",
        ]:
            if not budget_left():
                break
            print(f"[search] {q}")
            results.extend(search_web(q, num=5))
            search_count += 1

        extracted = extract_app(name, results)
        extracted["app_name"] = name
        extracted["source_url"] = app.get("url")
        extracted["_searched_at"] = run_meta["started_at"]
        new_apps.append(extracted)

    for q in DISCOVERY_QUERIES:
        if not budget_left():
            break
        print(f"[discovery] {q}")
        search_web(q, num=5)
        search_count += 1

    changes = []
    for app in new_apps:
        app_changes = diff_app(previous.get(app["app_name"]), app)
        if app_changes:
            changes.append({"app": app["app_name"], "changes": app_changes})

    run_meta["searches_used"] = search_count
    run_meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    run_meta["apps_tracked"] = len(new_apps)
    run_meta["changes_detected"] = len(changes)

    save_results(new_apps, run_meta)
    append_history(changes, run_meta)

    if changes:
        body = "\n".join(
            f"{c['app']}: {', '.join(c['changes'])}" for c in changes
        )
        notify_ntfy(f"DocGen Agent: {len(changes)} changes", body)
        print(f"[worker] Notified: {len(changes)} changes")
    else:
        print("[worker] No changes detected")

    print(f"[worker] Done. Searches: {search_count}/{MAX_SEARCHES_PER_RUN}")

if __name__ == "__main__":
    main()
