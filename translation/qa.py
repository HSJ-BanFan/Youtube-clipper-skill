from __future__ import annotations

import re
from dataclasses import dataclass

from translation.models import Cue


@dataclass(frozen=True)
class QAIssue:
    cue_id: str
    severity: str
    reason: str


@dataclass(frozen=True)
class QACandidate:
    cue: Cue
    translation: str
    issues: tuple[QAIssue, ...]


_URL_RE = re.compile(r"https?://\S+")
_TAG_RE = re.compile(r"</?[^>\s]+(?:\s+[^>]*)?>|<\d{2}:\d{2}:\d{2}[,.]\d{3}>")
_CODE_TOKEN_RE = re.compile(
    r"(?<!\w)(?:[A-Za-z]:\\[^\s]+|[\w./-]+\.(?:py|js|ts|json|md|srt|vtt|txt|yaml|yml)|--[\w-]+|[A-Z_]{2,}|[a-zA-Z_][\w-]*\([^)]*\)|`[^`]+`)(?!\w)"
)
_REFUSAL_MARKERS = (
    "as an ai",
    "i'm sorry",
    "i am sorry",
    "i cannot",
    "我不能",
    "作为ai",
    "作为 ai",
)
_BRACKET_PAIRS = (("(", ")"), ("[", "]"), ("{", "}"), ("（", "）"), ("【", "】"))


def find_suspicious_translations(
    cues: list[Cue],
    translations: dict[str, str],
    target_lang: str,
) -> list[QACandidate]:
    candidates: list[QACandidate] = []
    for cue in cues:
        translation = translations.get(cue.id, "")
        issues = _issues_for(cue, translation, target_lang)
        if issues:
            candidates.append(QACandidate(cue=cue, translation=translation, issues=tuple(issues)))
    return candidates


def _issues_for(cue: Cue, translation: str, target_lang: str) -> list[QAIssue]:
    source = cue.source
    stripped_translation = translation.strip()
    issues: list[QAIssue] = []

    if not stripped_translation:
        issues.append(_issue(cue.id, "high", "empty translation"))
        return issues

    lower_translation = stripped_translation.lower()
    if "```" in stripped_translation or _looks_like_json_leak(stripped_translation):
        issues.append(_issue(cue.id, "high", "json or markdown leak"))
    if any(marker in lower_translation for marker in _REFUSAL_MARKERS):
        issues.append(_issue(cue.id, "high", "model refusal"))
    if _count_urls(source) != _count_urls(stripped_translation):
        issues.append(_issue(cue.id, "medium", "url count mismatch"))
    if _count_tags(source) != _count_tags(stripped_translation):
        issues.append(_issue(cue.id, "medium", "tag count mismatch"))
    if _has_bracket_mismatch(source, stripped_translation):
        issues.append(_issue(cue.id, "medium", "bracket count mismatch"))
    if _has_missing_code_tokens(source, stripped_translation):
        issues.append(_issue(cue.id, "medium", "code token loss"))

    if _is_translatable_text(source) and _word_count(source) >= 10:
        source_len = len(source.strip())
        translation_len = len(stripped_translation)
        if translation_len < max(8, source_len * 0.18):
            issues.append(_issue(cue.id, "medium", "translation unusually short"))
        if translation_len > max(80, source_len * 3.5):
            issues.append(_issue(cue.id, "medium", "translation unusually expanded"))
        if _requires_target_script(target_lang) and _mostly_english(source) and not _has_target_script(stripped_translation, target_lang):
            issues.append(_issue(cue.id, "medium", "missing target-language characters"))

    return issues


def _issue(cue_id: str, severity: str, reason: str) -> QAIssue:
    return QAIssue(cue_id=cue_id, severity=severity, reason=reason)


def _looks_like_json_leak(text: str) -> bool:
    stripped = text.strip()
    if not ((stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]"))):
        return False
    return '"id"' in stripped or '"translation"' in stripped


def _count_urls(text: str) -> int:
    return len(_URL_RE.findall(text))


def _count_tags(text: str) -> int:
    return len(_TAG_RE.findall(text))


def _has_bracket_mismatch(source: str, translation: str) -> bool:
    for open_bracket, close_bracket in _BRACKET_PAIRS:
        source_count = source.count(open_bracket) + source.count(close_bracket)
        translation_count = translation.count(open_bracket) + translation.count(close_bracket)
        if source_count and abs(source_count - translation_count) >= 2:
            return True
    return False


def _has_missing_code_tokens(source: str, translation: str) -> bool:
    if _is_plain_url(source):
        return False
    source_tokens = set(_CODE_TOKEN_RE.findall(source))
    if not source_tokens:
        return False
    translation_tokens = set(_CODE_TOKEN_RE.findall(translation))
    missing = source_tokens - translation_tokens
    return len(missing) >= 2 or (len(source_tokens) == 1 and bool(missing))


def _is_translatable_text(source: str) -> bool:
    stripped = source.strip()
    if _is_plain_url(stripped):
        return False
    if _looks_like_command(stripped):
        return False
    letters = sum(character.isalpha() for character in stripped)
    return letters >= 12


def _is_plain_url(text: str) -> bool:
    return bool(_URL_RE.fullmatch(text.strip()))


def _looks_like_command(text: str) -> bool:
    if "\n" in text:
        return False
    parts = text.split()
    if len(parts) < 2:
        return False
    return any(part.startswith("--") for part in parts) or any("/" in part or "\\" in part for part in parts)


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+", text))


def _requires_target_script(target_lang: str) -> bool:
    return target_lang.lower() in {"zh", "zh-cn", "ja", "ko"}


def _mostly_english(text: str) -> bool:
    words = re.findall(r"[A-Za-z]+", text)
    return len(words) >= 8


def _has_target_script(text: str, target_lang: str) -> bool:
    lang = target_lang.lower()
    if lang in {"zh", "zh-cn", "ja"} and re.search(r"[一-鿿]", text):
        return True
    if lang == "ja" and re.search(r"[぀-ヿ]", text):
        return True
    if lang == "ko" and re.search(r"[가-힣]", text):
        return True
    return False
