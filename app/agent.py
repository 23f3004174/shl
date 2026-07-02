import re
from typing import Any, Dict, List, Optional, Sequence, Set
from app.retriever import get_retriever

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - fallback for environments without the package
    genai = None

from app.config import GEMINI_API_KEY


if genai is not None and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    _model = genai.GenerativeModel("gemini-2.5-flash")
else:
    _model = None


SYSTEM_PROMPT = """
You are an SHL Assessment Assistant.
Only answer questions about SHL assessments.
If the user asks unrelated questions, politely refuse.
If the user asks for assessment recommendations, use ONLY the retrieved assessments.
Never invent URLs.
Always return concise responses.
"""


LANGUAGE_RE = re.compile(r"\b(java|python|javascript|c#|c\+\+|\.net|typescript|sql|ruby|go|php)\b")
JOB_TITLE_RE = re.compile(
    r"\b(?:java|python|javascript|c#|c\+\+|\.net|typescript|sql|ruby|go|php)?\s*"
    r"(developer|engineer|manager|analyst|designer|specialist|consultant|architect|lead|executive|intern|graduate|support|sales|director|coordinator|administrator|technician)\b"
)
SENIORITY_RE = re.compile(r"\b(senior|mid[- ]level|junior|entry[- ]level|lead|principal|executive|graduate|intern)\b")
COMPARISON_RE = re.compile(r"\b(compare|difference between|difference)\b")

# Years-of-experience patterns, most specific first: ranges ("0-1 years"),
# open-ended ("7+ years"), then a bare number ("4 years").
_YEARS_RANGE_RE = re.compile(r"\b(\d+)\s*-\s*(\d+)\s*\+?\s*(?:years|yrs|year)\b")
_YEARS_PLUS_RE = re.compile(r"\b(\d+)\s*\+\s*(?:years|yrs|year)\b")
_YEARS_SINGLE_RE = re.compile(r"\b(\d+)\s*(?:years|yrs|year)\b")

UNRELATED_KEYWORDS = [
    "legal advice",
    "salary",
    "resume",
    "interview tips",
    "general hiring advice",
    "hiring advice",
    "hr policy",
    "prompt injection",
    "ignore previous instructions",
    "system prompt",
    "jailbreak",
    "political",
    "medical advice",
]

DOMAIN_KEYWORDS = {
    "sales",
    "customer support",
    "customer",
    "support",
    "leadership",
    "remote",
}

ASSESSMENT_TYPE_KEYWORDS = {
    "personality",
    "cognitive",
    "coding",
    "technical",
    "behavioral",
    "aptitude",
    "simulation",
    "skills",
}

# Any of these appearing anywhere in the full conversation text signals that
# the request is plausibly about SHL assessments / hiring. If NONE of these
# are present, the query is treated as out-of-domain (see _is_unrelated).
DOMAIN_RELEVANCE_KEYWORDS = {
    "assessment", "assessments", "test", "tests", "exam", "evaluate", "evaluation",
    "candidate", "candidates", "hire", "hiring", "recruit", "recruitment", "recruiting",
    "job", "role", "position", "skill", "skills", "personality", "cognitive", "aptitude",
    "technical", "coding", "programming", "behavioral", "behavioural", "simulation",
    "seniority", "experience", "years", "developer", "engineer", "manager", "analyst",
    "designer", "specialist", "consultant", "architect", "lead", "executive", "intern",
    "graduate", "sales", "support", "leadership", "remote", "java", "python", "javascript",
    "sql", "c#", "c++", ".net", "typescript", "ruby", "go", "php", "shl", "opq", "gsa",
    "compare", "comparison", "difference", "catalog", "catalogue",
}

# Common abbreviations / shorthand that should resolve to a fuller search
# phrase before hitting the retriever, so comparison queries like
# "Compare OPQ and GSA" don't require exact catalog-name matches.
ASSESSMENT_ALIASES = {
    "opq": "Occupational Personality Questionnaire OPQ32r",
    "gsa": "Global Skills Assessment",
    "java ee": "Java Platform Enterprise Edition",
    "java ee 7": "Java Platform Enterprise Edition 7",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _extract_user_messages(messages: Sequence[Any]) -> List[str]:
    extracted: List[str] = []
    for message in messages or []:
        if not isinstance(message, dict):
            role = getattr(message, "role", "")
            content = getattr(message, "content", "")
        else:
            role = message.get("role", "")
            content = message.get("content", "")

        if str(role).lower() == "user" and isinstance(content, str):
            extracted.append(content)

    return extracted


def _extract_all_user_text(messages: Sequence[Any]) -> str:
    return " ".join(_extract_user_messages(messages)).strip()


def _is_unrelated(query: str) -> bool:
    normalized = _normalize(query)
    if not normalized:
        return False

    if any(keyword in normalized for keyword in UNRELATED_KEYWORDS):
        return True

    # If the conversation (so far) contains no word that plausibly relates to
    # SHL assessments / hiring, treat it as out-of-domain. This catches
    # generic off-topic requests (weather, sports scores, legal help, etc.)
    # that don't match a specific blacklisted phrase above.
    has_domain_word = any(
        re.search(r"\b" + re.escape(word) + r"\b", normalized)
        for word in DOMAIN_RELEVANCE_KEYWORDS
    )
    return not has_domain_word


def _is_comparison_request(query: str) -> bool:
    q = _normalize(query)

    comparison_words = [
        "compare",
        "comparison",
        "difference",
        "different",
        "vs",
        "versus",
        "between"
    ]

    return any(word in q for word in comparison_words)


def _extract_years_info(normalized_text: str) -> Optional[Dict[str, Any]]:
    """Return {'raw': <matched text>, 'low': <lower-bound int>} for the first
    years-of-experience mention in normalized_text, or None if there isn't one."""

    range_match = _YEARS_RANGE_RE.search(normalized_text)
    if range_match:
        return {"raw": range_match.group(0).strip(), "low": int(range_match.group(1))}

    plus_match = _YEARS_PLUS_RE.search(normalized_text)
    if plus_match:
        return {"raw": plus_match.group(0).strip(), "low": int(plus_match.group(1))}

    single_match = _YEARS_SINGLE_RE.search(normalized_text)
    if single_match:
        return {"raw": single_match.group(0).strip(), "low": int(single_match.group(1))}

    return None


def _seniority_from_years(low: int) -> str:
    """Infer a seniority label from a lower-bound years-of-experience value.

    0-1 years -> entry-level, 2-6 years -> mid-level, 7+ years -> senior.
    """
    if low <= 1:
        return "entry-level"
    if low <= 6:
        return "mid-level"
    return "senior"


def _extract_constraints(messages: Sequence[Any]) -> Dict[str, Any]:
    constraints: Dict[str, Any] = {
        "role": None,
        "language": None,
        "job_title": None,
        "seniority": None,
        "years_experience": None,
        "assessment_types": set(),
        "domain": None,
        "extras": set(),
    }

    for content in _extract_user_messages(messages):
        normalized = _normalize(content)
        if not normalized:
            continue

        languages = LANGUAGE_RE.findall(normalized)
        if languages:
            constraints["language"] = languages[-1]

        role_match = JOB_TITLE_RE.search(normalized)
        if role_match:
            constraints["role"] = role_match.group(1)
            if constraints["role"] in {
                "developer",
                "engineer",
                "analyst",
                "designer",
                "specialist",
                "consultant",
                "architect",
                "technician",
            }:
                constraints["job_title"] = content.strip()

        seniority_match = SENIORITY_RE.search(normalized)
        if seniority_match:
            constraints["seniority"] = seniority_match.group(1)

        years_info = _extract_years_info(normalized)
        if years_info:
            constraints["years_experience"] = years_info["raw"]
            # Only infer seniority from years if it wasn't explicitly stated
            # (explicit wording like "senior" always takes priority).
            if not constraints["seniority"]:
                constraints["seniority"] = _seniority_from_years(years_info["low"])

        if "personality" in normalized:
            constraints["assessment_types"].add("personality")
        if "cognitive" in normalized or "reasoning" in normalized or "aptitude" in normalized:
            constraints["assessment_types"].add("cognitive")
        if "coding" in normalized or "programming" in normalized:
            constraints["assessment_types"].add("coding")
        if "technical" in normalized:
            constraints["assessment_types"].add("technical")
        if "behavioral" in normalized or "behavioural" in normalized:
            constraints["assessment_types"].add("behavioral")
        if "simulation" in normalized:
            constraints["assessment_types"].add("simulation")
        if "skills" in normalized and "assessment" in normalized:
            constraints["assessment_types"].add("skills")

        for domain in DOMAIN_KEYWORDS:
            if domain in normalized:
                constraints["domain"] = domain

        if "remote" in normalized:
            constraints["extras"].add("remote")
        if "leadership" in normalized:
            constraints["extras"].add("leadership")
        if "graduate" in normalized:
            constraints["extras"].add("graduate")
        if "manager" in normalized:
            constraints["role"] = "manager"
        if "executive" in normalized:
            constraints["role"] = "executive"

        if "developer" in normalized and not constraints["job_title"]:
            constraints["job_title"] = content.strip()
        if "engineer" in normalized and not constraints["job_title"]:
            constraints["job_title"] = content.strip()

    if not constraints["role"] and constraints["job_title"]:
        constraints["role"] = constraints["job_title"]

    return constraints


def _clarification_prompt(constraints):
    """
    Only clarify when we truly don't have enough information.
    """

    role = constraints["role"]
    language = constraints["language"]
    seniority = constraints["seniority"]
    years = constraints["years_experience"]
    assessment_types = constraints["assessment_types"]

    # absolutely nothing known
    if not role and not language:
        return (
            "What role are you hiring for? "
            "Please include the job role or technical skill."
        )

    # if role exists and ANY additional constraint exists,
    # recommend immediately.
    if role and (
        seniority
        or years
        or assessment_types
        or language
    ):
        return None

    # role only
    if role:
        return (
            "What seniority level is the role "
            "(entry, junior, mid-level or senior)?"
        )

    return None


def _build_retrieval_query(constraints: Dict[str, Any], fallback: str) -> str:
    pieces: List[str] = []
    if constraints["role"]:
        pieces.append(str(constraints["role"]))
    if constraints["language"]:
        pieces.append(str(constraints["language"]))
    if constraints["seniority"]:
        pieces.append(str(constraints["seniority"]))
    if constraints["years_experience"]:
        pieces.append(f"{constraints['years_experience']} years")
    if constraints["domain"]:
        pieces.append(str(constraints["domain"]))
    if constraints["assessment_types"]:
        pieces.extend(sorted(constraints["assessment_types"]))
    if constraints["extras"]:
        pieces.extend(sorted(constraints["extras"]))

    query = " ".join(pieces).strip()
    return query if query else fallback


def _build_recommendations(docs: List[dict]) -> List[dict]:
    recommendations = []
    for doc in docs[:10]:
        if not doc:
            continue
        recommendations.append(
            {
                "name": doc.get("name", "Unknown assessment"),
                "url": doc.get("url", ""),
                "test_type": doc.get("test_type") or doc.get("type") or "Assessment",
            }
        )
    return recommendations


def _comparison_terms(query: str) -> List[str]:
    """Extract the two things being compared from a comparison query.

    Trims off leading question phrasing ("what is the difference between...")
    so we don't accidentally grab the question stem as one of the terms, then
    splits the remainder on "and" / "vs" / "versus" / commas.
    """

    lower = query.lower()
    start = 0
    for marker in ("difference between", "between", "compare"):
        idx = lower.find(marker)
        if idx != -1:
            start = idx + len(marker)
            break

    segment = query[start:].strip(" ?.!\u2013\u2014")
    parts = re.split(r"\s+(?:and|vs\.?|versus|,)\s+", segment, flags=re.IGNORECASE)
    return [part.strip(" ?.!") for part in parts if part.strip(" ?.!") and len(part.strip()) > 1]


def _resolve_alias(term: str) -> str:
    return ASSESSMENT_ALIASES.get(_normalize(term), term)


def _select_comparison_match(docs: List[dict], expanded_term: str) -> Optional[dict]:
    """Pick the best catalog entry for a comparison term out of the
    retriever's ranked results.

    Prefer an entry whose NAME contains the expanded alias (e.g. picking
    "Occupational Personality Questionnaire OPQ32r" over an unrelated
    "OPQ Manager Plus Report 2.0" that merely also matched "OPQ"). If no
    entry's name contains the expanded term, fall back to the highest-ranked
    retriever result.
    """

    if not docs:
        return None

    expanded_norm = _normalize(expanded_term)
    if expanded_norm:
        for doc in docs:
            if expanded_norm in _normalize(doc.get("name", "")):
                return doc

    return docs[0]


def _comparison_reply(query: str) -> str:
    terms = _comparison_terms(query)

    if len(terms) < 2:
        return "I can compare SHL assessments from the catalog when you name the two assessments you want to compare."

    matched_docs = []
    for term in terms[:2]:
        search_term = _resolve_alias(term)
        try:
            docs = get_retriever().search(search_term, top_k=5)
        except Exception:
            docs = []
        # Prefer a catalog entry whose name contains the expanded alias;
        # fall back to the retriever's top-ranked result otherwise.
        match = _select_comparison_match(docs, search_term)
        if match:
            matched_docs.append(match)

    if len(matched_docs) < 2:
        return "I do not have enough catalog information to compare those assessments reliably."

    comparisons = [
        f"{doc.get('name', 'Unknown assessment')}: {doc.get('description', 'No catalog description available.')}"
        for doc in matched_docs
    ]

    return "Based on the catalog entries I can access: " + " | ".join(comparisons)


def _find_comparison_source_text(messages: Sequence[Any], fallback: str) -> str:
    """Prefer the specific user message that actually asked for a comparison,
    rather than the entire concatenated conversation history, so unrelated
    earlier turns don't pollute term extraction."""

    for content in reversed(_extract_user_messages(messages)):
        if _is_comparison_request(content):
            return content
    return fallback




def chat(messages: Sequence[Any]):
    """Process the full conversation history and return a schema-safe response payload."""

    if not messages:
        return "Please share the role, seniority, or assessment type you need.", [], False

    full_query = _extract_all_user_text(messages)
    if not full_query:
        return "Please send a non-empty user message.", [], False

    if _is_unrelated(full_query):
        return (
            "I can only help with SHL assessment recommendations and catalog-based questions.",
            [],
            False,
        )

    if _is_comparison_request(full_query):
        comparison_source = _find_comparison_source_text(messages, full_query)
        return _comparison_reply(comparison_source), [], False

    constraints = _extract_constraints(messages)
    
    # Merge information from the entire conversation
# before asking again.

    if constraints["job_title"] and not constraints["role"]:
        constraints["role"] = constraints["job_title"]
    
    
    clarification = _clarification_prompt(constraints)
    if clarification is not None:
        return clarification, [], False
    
    query = (
    full_query
    + " "
    + _build_retrieval_query(constraints, full_query)
)
    try:
        docs = get_retriever().search(query, top_k=5)
    except Exception:
        return (
            "I could not retrieve assessments right now. Please try again later.",
            [],
            False,
        )

    if not docs:
        return (
            "I could not find SHL assessments matching your request. Please refine your requirements.",
            [],
            False,
        )

    recommendations = _build_recommendations(docs)

    if _model is not None:
        context = "\n".join(
            f"{doc.get('name', '')}: {doc.get('description', '')}" for doc in docs[:5]
        )
        prompt = SYSTEM_PROMPT + "\n\nCatalog context:\n" + context + "\n\nUser:\n" + full_query
        try:
            response = _model.generate_content(prompt)
            reply = response.text
        except Exception:
            reply = "Here are a few SHL assessments that match your request."
    else:
        reply = "Here are a few SHL assessments that match your request."

    return reply, recommendations, True