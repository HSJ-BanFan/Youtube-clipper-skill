from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
