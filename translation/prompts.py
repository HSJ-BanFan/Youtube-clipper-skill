from __future__ import annotations

import json
from collections.abc import Sequence

from translation.models import Cue, TranslationBatch
from translation.qa import QACandidate


PROMPT_VERSION = "translation-v2-json-cue-v1"
QA_PROMPT_VERSION = "translation-v2-suspicious-qa-v1"


def build_translation_prompt(
    batch: TranslationBatch,
    target_lang: str,
    glossary_text: str = "",
    global_context_text: str = "",
) -> str:
    return "\n".join(
        [
            f"Translate current subtitle cues into {target_lang}.",
            "Technical subtitles: prioritize accuracy, then natural concise wording.",
            "only return JSON. do not return Markdown or explanations.",
            "Return a JSON array whose item count must equal the number of cues.",
            "Each item must have id and translation fields.",
            "id must match exactly. translation must not be empty.",
            "preserve code, commands, variable names, paths, URLs, and library names.",
            "Follow the glossary consistently when translating current cues.",
            "Global context is only for understanding; do not translate global context or output it.",
            "Before/after context is only for understanding; do not translate context or include context ids in output.",
            "Expected JSON shape:",
            '[{"id": "1", "translation": "..."}]',
            "Glossary:",
            _format_text_section(glossary_text),
            "Global Context:",
            _format_text_section(global_context_text),
            "Before context:",
            _format_cues(batch.context_before),
            "Current cues to translate:",
            _format_cues(batch.cues),
            "After context:",
            _format_cues(batch.context_after),
        ]
    )


def build_suspicious_qa_prompt(
    candidates: list[QACandidate],
    target_lang: str,
    glossary_text: str = "",
    global_context_text: str = "",
) -> str:
    return "\n".join(
        [
            f"Review only suspicious subtitle translations for {target_lang}.",
            "Only fix obvious errors; otherwise keep the existing translation.",
            "only return JSON. do not return Markdown or explanations.",
            "Return a JSON array with exactly one item for each candidate.",
            "do not add, delete, or reorder ids. id must match exactly.",
            'Each item shape: {"id": "...", "action": "keep" | "fix", "translation": "...", "reason": "..."}.',
            "translation must not be empty.",
            "preserve code, commands, variable names, paths, URLs, and library names.",
            "Follow the glossary consistently when fixing translations.",
            "Global context is only for understanding; do not translate global context; do not output global context.",
            "Glossary:",
            _format_text_section(glossary_text),
            "Global Context:",
            _format_text_section(global_context_text),
            "Suspicious candidates:",
            _format_qa_candidates(candidates),
        ]
    )


def _format_text_section(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "[]"
    return stripped


def _format_cues(cues: Sequence[Cue]) -> str:
    if not cues:
        return "[]"
    return json.dumps(
        [{"id": cue.id, "source": cue.source} for cue in cues],
        ensure_ascii=False,
        indent=2,
    )


def _format_qa_candidates(candidates: Sequence[QACandidate]) -> str:
    if not candidates:
        return "[]"
    return json.dumps(
        [
            {
                "id": candidate.cue.id,
                "source": candidate.cue.source,
                "translation": candidate.translation,
                "issues": [
                    {
                        "severity": issue.severity,
                        "reason": issue.reason,
                    }
                    for issue in candidate.issues
                ],
            }
            for candidate in candidates
        ],
        ensure_ascii=False,
        indent=2,
    )
