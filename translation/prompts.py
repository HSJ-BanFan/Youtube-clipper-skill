from __future__ import annotations

import json
from collections.abc import Sequence

from translation.models import Cue, TranslationBatch


def build_translation_prompt(batch: TranslationBatch, target_lang: str) -> str:
    return "\n".join(
        [
            f"Translate current subtitle cues into {target_lang}.",
            "Technical subtitles: prioritize accuracy, then natural concise wording.",
            "only return JSON. do not return Markdown or explanations.",
            "Return a JSON array whose item count must equal the number of cues.",
            "Each item must have id and translation fields.",
            "id must match exactly. translation must not be empty.",
            "preserve code, commands, variable names, paths, URLs, and library names.",
            "Before/after context is only for understanding; do not translate context or include context ids in output.",
            "Expected JSON shape:",
            '[{"id": "1", "translation": "..."}]',
            "Before context:",
            _format_cues(batch.context_before),
            "Current cues to translate:",
            _format_cues(batch.cues),
            "After context:",
            _format_cues(batch.context_after),
        ]
    )


def _format_cues(cues: Sequence[Cue]) -> str:
    if not cues:
        return "[]"
    return json.dumps(
        [{"id": cue.id, "source": cue.source} for cue in cues],
        ensure_ascii=False,
        indent=2,
    )
