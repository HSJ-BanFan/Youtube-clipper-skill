from __future__ import annotations

import re
from collections import Counter
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
_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:\\[^\s]+")
_POSIX_PATH_RE = re.compile(r"(?<![A-Za-z0-9])/(?:[\w.-]+/)+[\w.-]+")
_OPTION_RE = re.compile(r"--[\w-]+")
_ENV_VAR_RE = re.compile(r"(?<!\w)[A-Z][A-Z0-9_]{2,}(?!\w)")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_NUMBER_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(ms|millisecond(?:s)?|sec(?:ond)?(?:s)?|s|minute(?:s)?|hour(?:s)?|gb|mb|kb|tb|px|fps|hz|khz|mhz|ghz|%|x|毫秒|秒|分钟|小时)",
    re.IGNORECASE,
)
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
_POLLUTED_OUTPUT_PREFIXES = (
    "translation:",
    "translated:",
    "explanation:",
    "note:",
    "here is",
)
_TECHNICAL_MARKERS = (
    "ffmpeg",
    "codec",
    "api",
    "json",
    "yaml",
    "python",
    "cli",
    "subtitle",
    "script",
    "command",
    "token",
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
    if _looks_like_polluted_output(lower_translation):
        issues.append(_issue(cue.id, "medium", "polluted output"))
    if _has_url_mismatch(source, stripped_translation):
        issues.append(_issue(cue.id, "medium", "url count mismatch"))
    if _has_numeric_mismatch(source, stripped_translation):
        issues.append(_issue(cue.id, "medium", "numeric token mismatch"))
    if _has_unit_mismatch(source, stripped_translation):
        issues.append(_issue(cue.id, "medium", "unit token mismatch"))
    if _has_path_mismatch(source, stripped_translation):
        issues.append(_issue(cue.id, "medium", "path token mismatch"))
    if _has_option_mismatch(source, stripped_translation):
        issues.append(_issue(cue.id, "medium", "option token mismatch"))
    if _has_env_var_mismatch(source, stripped_translation):
        issues.append(_issue(cue.id, "medium", "env var mismatch"))
    if _count_tags(source) != _count_tags(stripped_translation):
        issues.append(_issue(cue.id, "medium", "tag count mismatch"))
    if _has_bracket_mismatch(source, stripped_translation):
        issues.append(_issue(cue.id, "medium", "bracket count mismatch"))
    if _has_missing_code_tokens(source, stripped_translation):
        issues.append(_issue(cue.id, "medium", "code token loss"))

    source_len = len(source.strip())
    translation_len = len(stripped_translation)
    if _is_translatable_text(source) and _word_count(source) >= 10:
        if translation_len < max(8, source_len * 0.18):
            issues.append(_issue(cue.id, "medium", "translation unusually short"))
        if translation_len > max(80, source_len * 3.5):
            issues.append(_issue(cue.id, "medium", "translation unusually expanded"))
        if _requires_target_script(target_lang) and _mostly_english(source) and not _has_target_script(stripped_translation, target_lang):
            issues.append(_issue(cue.id, "medium", "missing target-language characters"))
    elif _is_short_technical_text(source) and translation_len < max(4, source_len * 0.18):
        issues.append(_issue(cue.id, "medium", "translation unusually short"))

    return issues


def _issue(cue_id: str, severity: str, reason: str) -> QAIssue:
    return QAIssue(cue_id=cue_id, severity=severity, reason=reason)


def _looks_like_json_leak(text: str) -> bool:
    stripped = text.strip()
    if not ((stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]"))):
        return False
    return '"id"' in stripped or '"translation"' in stripped


def _has_url_mismatch(source: str, translation: str) -> bool:
    source_urls = _URL_RE.findall(source)
    if not source_urls:
        return False
    return _has_token_mismatch(source_urls, _URL_RE.findall(translation))


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
    source_tokens = _CODE_TOKEN_RE.findall(source)
    if not source_tokens:
        return False
    return _has_token_mismatch(source_tokens, _CODE_TOKEN_RE.findall(translation))


def _has_numeric_mismatch(source: str, translation: str) -> bool:
    source_numbers = _NUMBER_RE.findall(source)
    if not source_numbers:
        return False
    return _has_token_mismatch(source_numbers, _NUMBER_RE.findall(translation))


def _has_unit_mismatch(source: str, translation: str) -> bool:
    source_units = _normalized_number_units(source)
    if not source_units:
        return False
    return _has_token_mismatch(source_units, _normalized_number_units(translation))


def _has_path_mismatch(source: str, translation: str) -> bool:
    source_paths = _extract_paths(source)
    if not source_paths:
        return False
    return _has_token_mismatch(source_paths, _extract_paths(translation))


def _has_option_mismatch(source: str, translation: str) -> bool:
    source_options = _OPTION_RE.findall(source)
    if not source_options:
        return False
    return _has_token_mismatch(source_options, _OPTION_RE.findall(translation))


def _has_env_var_mismatch(source: str, translation: str) -> bool:
    source_env_vars = _ENV_VAR_RE.findall(source)
    if not source_env_vars:
        return False
    return _has_token_mismatch(source_env_vars, _ENV_VAR_RE.findall(translation))


def _looks_like_polluted_output(lower_translation: str) -> bool:
    return any(lower_translation.startswith(prefix) for prefix in _POLLUTED_OUTPUT_PREFIXES)


def _has_token_mismatch(source_tokens: list[str] | list[tuple[str, str]], translation_tokens: list[str] | list[tuple[str, str]]) -> bool:
    return Counter(source_tokens) != Counter(translation_tokens)


def _normalized_number_units(text: str) -> list[tuple[str, str]]:
    return [
        (number, _normalize_unit(unit))
        for number, unit in _NUMBER_UNIT_RE.findall(text)
    ]


def _normalize_unit(unit: str) -> str:
    normalized = unit.lower()
    if normalized in {"ms", "millisecond", "milliseconds", "毫秒"}:
        return "ms"
    if normalized in {"s", "sec", "second", "seconds", "秒"}:
        return "s"
    if normalized in {"minute", "minutes", "分钟"}:
        return "min"
    if normalized in {"hour", "hours", "小时"}:
        return "h"
    return normalized


def _extract_paths(text: str) -> list[str]:
    return _WINDOWS_PATH_RE.findall(text) + _POSIX_PATH_RE.findall(text)


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


def _is_short_technical_text(source: str) -> bool:
    lowered = source.lower()
    return _word_count(source) >= 5 and (
        bool(_CODE_TOKEN_RE.search(source))
        or bool(_NUMBER_RE.search(source))
        or any(marker in lowered for marker in _TECHNICAL_MARKERS)
    )


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
