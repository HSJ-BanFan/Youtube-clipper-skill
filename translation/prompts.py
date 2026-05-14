from __future__ import annotations

import json
from collections.abc import Sequence

from translation.models import BatchRecord, Cue, CueRecord, TranslationBatch
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


def build_structured_translation_prompt(
    batch: TranslationBatch,
    target_lang: str,
    glossary_text: str = "",
    global_context_text: str = "",
    batch_record: BatchRecord | None = None,
) -> str:
    before_context = _format_cue_records(batch_record.context_before) if batch_record is not None else _format_cues(batch.context_before, cue_key="cue_id")
    current_cues = _format_cue_records(batch_record.target_cues) if batch_record is not None else _format_cues(batch.cues, cue_key="cue_id")
    after_context = _format_cue_records(batch_record.context_after) if batch_record is not None else _format_cues(batch.context_after, cue_key="cue_id")
    return "\n".join(
        [
            f"Translate current subtitle cues into {target_lang}.",
            "Technical subtitles: prioritize accuracy, then natural concise wording.",
            "only return JSON. do not return Markdown or explanations.",
            "Return a JSON array whose item count must equal the number of cues.",
            "Each item must have cue_id and translation fields.",
            "cue_id must match exactly. translation must not be empty.",
            "preserve code, commands, variable names, paths, URLs, and library names.",
            "Follow the glossary consistently when translating current cues.",
            "Global context is only for understanding; do not translate global context or output it.",
            "Before/after context is only for understanding; do not translate context or include context ids in output.",
            "Expected JSON shape:",
            '[{"cue_id": "1", "translation": "..."}]',
            "Glossary:",
            _format_text_section(glossary_text),
            "Global Context:",
            _format_text_section(global_context_text),
            "Before context:",
            before_context,
            "Current cues to translate:",
            current_cues,
            "After context:",
            after_context,
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


def _format_cues(cues: Sequence[Cue], cue_key: str = "id") -> str:
    if not cues:
        return "[]"
    return json.dumps(
        [{cue_key: cue.id, "source": cue.source} for cue in cues],
        ensure_ascii=False,
        indent=2,
    )


def _format_cue_records(cues: Sequence[CueRecord]) -> str:
    if not cues:
        return "[]"
    return json.dumps(
        [{"cue_id": cue.cue_id, "source": cue.source_text} for cue in cues],
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
