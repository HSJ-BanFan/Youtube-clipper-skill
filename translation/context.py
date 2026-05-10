from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from translation.config import TranslationConfig
from translation.models import Cue


SAMPLE_SIZE = 5


@dataclass(frozen=True)
class GlobalContext:
    text: str
    hash: str


def build_global_context(cues: list[Cue], input_path: Path, config: TranslationConfig) -> GlobalContext:
    text = _render_context(cues, input_path, config)
    return GlobalContext(
        text=text,
        hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def write_global_context(context: GlobalContext, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(context.text, encoding="utf-8")


def _render_context(cues: list[Cue], input_path: Path, config: TranslationConfig) -> str:
    sections = [
        "# Translation Global Context",
        "",
        "## Metadata",
        f"- input_filename: {input_path.name}",
        f"- cue_count: {len(cues)}",
        f"- target_lang: {config.target_lang}",
        f"- mode: {config.mode}",
        f"- batch_size: {config.batch_size}",
        "",
        "## Front Samples",
        *_format_sample_cues(_front_samples(cues)),
        "",
        "## Middle Samples",
        *_format_sample_cues(_middle_samples(cues)),
        "",
        "## Tail Samples",
        *_format_sample_cues(_tail_samples(cues)),
        "",
    ]
    return "\n".join(sections)


def _front_samples(cues: list[Cue]) -> list[Cue]:
    return cues[:SAMPLE_SIZE]


def _middle_samples(cues: list[Cue]) -> list[Cue]:
    if not cues:
        return []
    start = max((len(cues) - SAMPLE_SIZE) // 2, 0)
    return cues[start : start + SAMPLE_SIZE]


def _tail_samples(cues: list[Cue]) -> list[Cue]:
    if not cues:
        return []
    return cues[-SAMPLE_SIZE:]


def _format_sample_cues(cues: list[Cue]) -> list[str]:
    if not cues:
        return ["- none"]
    return [f"- [{cue.index}] {cue.source}" for cue in cues]
