from __future__ import annotations

import re
from pathlib import Path

from translation.config import TranslationConfig
from translation.models import PipelineResult, TranslationOutputPaths


def run_translation_pipeline(subtitle_path: str | Path, config: TranslationConfig) -> PipelineResult:
    input_path = Path(subtitle_path)
    if not input_path.exists():
        raise FileNotFoundError(f"subtitle_path not found: {input_path}")

    input_format = _detect_input_format(input_path)
    output_paths = build_output_paths(input_path, config)
    _ensure_outputs_do_not_exist(output_paths, config.overwrite)

    cue_count = _count_cues(input_path, input_format)

    if config.dry_run:
        return PipelineResult(
            input_path=input_path,
            input_format=input_format,
            output_format="srt",
            output_paths=output_paths,
            dry_run=True,
            cue_count=cue_count,
            provider_called=False,
        )

    raise NotImplementedError(
        "Translation provider execution is not implemented in this config/CLI section. "
        "Use --dry-run for configuration validation."
    )


def build_output_paths(input_path: Path, config: TranslationConfig) -> TranslationOutputPaths:
    output_dir = Path(config.output_dir) if config.output_dir else input_path.parent / "translated"
    return TranslationOutputPaths(
        output_dir=output_dir,
        translated_srt=output_dir / f"translated.{config.target_lang}.srt",
        bilingual_srt=output_dir / "bilingual.srt",
        translation_report=output_dir / "translation_report.md",
        global_context=output_dir / "global_context.md",
    )


def _ensure_outputs_do_not_exist(paths: TranslationOutputPaths, overwrite: bool) -> None:
    if overwrite:
        return
    candidates = [paths.translated_srt, paths.bilingual_srt, paths.translation_report, paths.global_context]
    existing = [path for path in candidates if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output file already exists: {joined}. Add --overwrite to replace outputs.")


def _detect_input_format(input_path: Path) -> str:
    suffix = input_path.suffix.lower()
    if suffix == ".srt":
        return "srt"
    if suffix == ".vtt":
        return "vtt"
    raise ValueError("subtitle_path must point to .srt or .vtt")


def _count_cues(input_path: Path, input_format: str) -> int:
    content = input_path.read_text(encoding="utf-8-sig")
    if input_format == "srt":
        return len(re.findall(r"(?m)^\d+\s*$", content))
    return content.count("-->")
