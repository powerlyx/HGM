"""
Plain-text prompt templates, section-marker parsers, and validation logic
for the robust A-MEM system. Replaces JSON-schema LLM calls with plain-text
prompts that work with any LLM backend (Ollama, SGLang, OpenAI, etc.).
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional, Callable

logger = logging.getLogger("amem_robust")

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences from LLM output."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?\s*```$', '', text, flags=re.MULTILINE)
    return text.strip()


def parse_with_json_fallback(response: str, plain_text_parser: Callable, *parser_args) -> Any:
    """Try JSON parsing first; fall back to section-marker parsing."""
    try:
        cleaned = strip_markdown_fences(response)
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return plain_text_parser(response, *parser_args)


# ---------------------------------------------------------------------------
# List parsing helpers
# ---------------------------------------------------------------------------

def _parse_list_items(text: str) -> List[str]:
    """Parse a section of text into a list of items."""
    if not text or not text.strip():
        return []

    lines = text.strip().splitlines()
    items: List[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^[\-\*•]\s*', '', line)
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        line = line.strip().strip('"').strip("'").strip()
        if not line:
            continue
        if ',' in line:
            for part in line.split(','):
                part = part.strip().strip('"').strip("'").strip()
                if part:
                    items.append(part)
        else:
            items.append(line)

    return items


def _extract_section(text: str, marker: str, next_markers: Optional[List[str]] = None) -> str:
    """Extract the text between *marker*: and the next known marker (or end)."""
    pattern = re.compile(
        rf'^\s*{re.escape(marker)}\s*:\s*(.*)$',
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return ""

    start = match.end()
    first_line = match.group(1).strip()

    end = len(text)
    if next_markers:
        for nm in next_markers:
            nm_pattern = re.compile(
                rf'^\s*{re.escape(nm)}\s*:', re.IGNORECASE | re.MULTILINE
            )
            nm_match = nm_pattern.search(text, start)
            if nm_match and nm_match.start() < end:
                end = nm_match.start()

    rest = text[start:end].strip()
    if first_line and rest:
        return first_line + "\n" + rest
    return first_line or rest


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ANALYZE_CONTENT_PROMPT = """Analyze the following content and provide:
1. KEYWORDS: The most important keywords (nouns, verbs, key concepts). Order from most to least important. At least three keywords. Do not include speaker names or time references.
2. CONTEXT: One sentence summarizing the main topic, key points, and purpose.
3. TAGS: Broad categories/themes for classification (domain, format, type). At least three tags.

Respond using EXACTLY this format (one section per header):

KEYWORDS: keyword1, keyword2, keyword3, ...
CONTEXT: A single sentence summarizing the content.
TAGS: tag1, tag2, tag3, ...

Content for analysis:
{content}"""


FOCUSED_KEYWORDS_PROMPT = """List exactly 5 keywords that capture the main concepts of the following text. Output only the keywords, comma-separated, nothing else.

Text: {content}"""


# ---------------------------------------------------------------------------
# Parsers for each call site
# ---------------------------------------------------------------------------

def parse_analyze_content(response: str, content: str = "") -> Dict[str, Any]:
    def _section_parse(resp: str, content_text: str = "") -> Dict[str, Any]:
        keywords_text = _extract_section(resp, "KEYWORDS", ["CONTEXT", "TAGS"])
        context_text = _extract_section(resp, "CONTEXT", ["TAGS", "KEYWORDS"])
        tags_text = _extract_section(resp, "TAGS", ["KEYWORDS", "CONTEXT"])

        keywords = _parse_list_items(keywords_text)
        context = context_text.strip() if context_text.strip() else ""
        tags = _parse_list_items(tags_text)

        return {"keywords": keywords, "context": context, "tags": tags}

    result = parse_with_json_fallback(response, _section_parse, content)
    result = validate_analysis_result(result, content)
    return result


def parse_plain_text_answer(response: str) -> str:
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "answer" in data:
            response = str(data["answer"])
    except (json.JSONDecodeError, ValueError):
        pass

    cleaned = strip_markdown_fences(response).strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        cleaned = lines[0]

    cleaned = re.sub(
        r'^(?:answer|short answer|final answer|response|prediction)\s*:\s*',
        '',
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip().strip('"').strip("'").strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned


def parse_relevant_parts(response: str) -> str:
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "relevant_parts" in data:
            return str(data["relevant_parts"])
    except (json.JSONDecodeError, ValueError):
        pass
    return strip_markdown_fences(response).strip()


def parse_keywords_response(response: str) -> str:
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "keywords" in data:
            return str(data["keywords"])
    except (json.JSONDecodeError, ValueError):
        pass
    return strip_markdown_fences(response).strip()


# ---------------------------------------------------------------------------
# Validation / heuristic repair
# ---------------------------------------------------------------------------

def validate_analysis_result(result: Dict[str, Any], content: str = "") -> Dict[str, Any]:
    if not isinstance(result, dict):
        result = {"keywords": [], "context": "", "tags": []}

    keywords = result.get("keywords", [])
    context = result.get("context", "")
    tags = result.get("tags", [])

    if isinstance(keywords, str):
        keywords = _parse_list_items(keywords)
    if isinstance(tags, str):
        tags = _parse_list_items(tags)
    if isinstance(context, list):
        context = " ".join(context)

    if not keywords and content:
        keywords = _heuristic_keywords(content)

    if not context and content:
        context = _heuristic_context(content)

    if not tags and keywords:
        tags = keywords[:3]

    result["keywords"] = keywords
    result["context"] = context
    result["tags"] = tags
    return result


def _heuristic_keywords(content: str, max_keywords: int = 5) -> List[str]:
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
        'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'out', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
        'if', 'while', 'about', 'up', 'it', 'its', 'i', 'me', 'my',
        'you', 'your', 'he', 'she', 'they', 'we', 'this', 'that', 'these',
        'those', 'what', 'which', 'who', 'whom', 'says', 'said', 'speaker',
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', content)
    scored = []
    seen = set()
    for w in words:
        w_lower = w.lower()
        if w_lower in stop_words or w_lower in seen:
            continue
        seen.add(w_lower)
        score = 2 if w[0].isupper() else 1
        scored.append((w_lower, score))

    scored.sort(key=lambda x: -x[1])
    return [w for w, _ in scored[:max_keywords]]


def _heuristic_context(content: str) -> str:
    match = re.match(r'(.+?[.!?])\s', content)
    if match:
        return match.group(1).strip()
    return content[:200].strip()

# ---------------------------------------------------------------------------
# Batched turn analysis: one LLM call for many turn contents
# ---------------------------------------------------------------------------

ANALYZE_CONTENT_BATCH_PROMPT = """Analyze each of the {n} text snippets below. For EACH snippet (in order), produce an object with:
- "keywords": at least three important keywords (nouns, verbs, key concepts), most important first. Do NOT include speaker names or time references.
- "context": one sentence summarizing the main topic, key points and purpose.
- "tags": at least three broad categories/themes (domain, format, type).

Output ONLY a JSON array of {n} objects (element i corresponds to snippet [i]), no prose, no markdown fences. Example schema:
[{{"keywords":["k1","k2","k3"],"context":"one sentence.","tags":["t1","t2","t3"]}}, ...]

Snippets:
{snippets}"""


def parse_analyze_content_batch(response: str, n: int, contents: List[str]) -> List[Dict[str, Any]]:
    """Parse a batched analysis response into a list of n analysis dicts.

    Robust to partial/ malformed output: any missing or invalid element falls
    back to heuristic analysis for that position (never raises).
    """
    import json as _json
    fallbacks = [
        {"keywords": _heuristic_keywords(c), "context": _heuristic_context(c), "tags": _heuristic_keywords(c)[:3]}
        for c in contents
    ]
    if not response or not str(response).strip():
        return fallbacks

    text = str(response).strip()
    # strip markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    parsed: Optional[List[Any]] = None
    try:
        obj = _json.loads(text)
        if isinstance(obj, list):
            parsed = obj
    except Exception:
        # last resort: locate the first '[' ... matching ']' substring
        start = text.find("[")
        end = text.rfind("]")
        if 0 <= start < end:
            try:
                cand = _json.loads(text[start:end + 1])
                if isinstance(cand, list):
                    parsed = cand
            except Exception:
                parsed = None

    results: List[Dict[str, Any]] = []
    for i in range(n):
        fb = fallbacks[i] if i < len(fallbacks) else {"keywords": [], "context": "", "tags": []}
        elem = parsed[i] if parsed and i < len(parsed) else None
        if not isinstance(elem, dict):
            results.append(fb)
            continue
        keywords = elem.get("keywords", [])
        context = elem.get("context", "")
        tags = elem.get("tags", [])
        ann = validate_analysis_result(
            {"keywords": keywords, "context": context, "tags": tags},
            contents[i] if i < len(contents) else "",
        )
        results.append(ann)
    return results

