from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Cue:
    id: str
    index: int
    start: str
    end: str
    source: str
    raw_timing: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class TranslatedCue:
    cue: Cue
    translation: str


@dataclass(frozen=True)
class TranslationOutputPaths:
    output_dir: Path
    translated_srt: Path
    bilingual_srt: Path
    translation_report: Path
    global_context: Path


@dataclass(frozen=True)
class PipelineResult:
    input_path: Path
    input_format: str
    output_format: str
    output_paths: TranslationOutputPaths
    dry_run: bool
    cue_count: int
    provider_called: bool = False
    first_cue_preview: str | None = None
    last_cue_preview: str | None = None
