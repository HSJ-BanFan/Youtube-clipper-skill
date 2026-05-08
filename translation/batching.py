from __future__ import annotations

from translation.models import Cue, TranslationBatch


def create_batches(
    cues: list[Cue],
    batch_size: int,
    context_before: int,
    context_after: int,
) -> list[TranslationBatch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if context_before < 0 or context_after < 0:
        raise ValueError("context_before and context_after must be greater than or equal to 0")

    batches: list[TranslationBatch] = []
    for start in range(0, len(cues), batch_size):
        end = min(start + batch_size, len(cues))
        before_start = max(0, start - context_before)
        after_end = min(len(cues), end + context_after)
        batches.append(
            TranslationBatch(
                batch_id=len(batches) + 1,
                cues=tuple(cues[start:end]),
                context_before=tuple(cues[before_start:start]),
                context_after=tuple(cues[end:after_end]),
            )
        )
    return batches
