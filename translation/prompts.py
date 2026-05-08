from __future__ import annotations

import json
from collections.abc import Sequence

from translation.models import Cue, TranslationBatch


PROMPT_VERSION = "translation-v2-json-cue-v1"


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
