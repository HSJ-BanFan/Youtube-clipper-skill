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


def allocate_child_batch_ids(next_child_batch_id: int) -> tuple[int, int, int]:
    if next_child_batch_id <= 0:
        raise ValueError("next_child_batch_id must be greater than 0")
    left_child_id = next_child_batch_id
    right_child_id = next_child_batch_id + 1
    return left_child_id, right_child_id, next_child_batch_id + 2



def split_batch(
    batch: TranslationBatch,
    left_child_id: int,
    right_child_id: int,
) -> tuple[TranslationBatch, TranslationBatch]:
    if len(batch.cues) < 2:
        raise ValueError("cannot split batch with fewer than 2 target cues")
    if left_child_id == right_child_id:
        raise ValueError("child batch ids must be unique")

    midpoint = len(batch.cues) // 2
    left_child = TranslationBatch(
        batch_id=left_child_id,
        cues=batch.cues[:midpoint],
        context_before=batch.context_before,
        context_after=batch.context_after,
    )
    right_child = TranslationBatch(
        batch_id=right_child_id,
        cues=batch.cues[midpoint:],
        context_before=batch.context_before,
        context_after=batch.context_after,
    )
    _validate_split_children(batch, left_child, right_child)
    return left_child, right_child



def _validate_split_children(
    parent_batch: TranslationBatch,
    left_child: TranslationBatch,
    right_child: TranslationBatch,
) -> None:
    left_ids = {cue.id for cue in left_child.cues}
    right_ids = {cue.id for cue in right_child.cues}
    if not left_ids.isdisjoint(right_ids):
        raise ValueError("child target cues must not overlap")

    parent_ids = tuple(cue.id for cue in parent_batch.cues)
    child_ids = tuple(cue.id for cue in left_child.cues + right_child.cues)
    if child_ids != parent_ids:
        raise ValueError("child target cues must preserve parent cue union and order")
