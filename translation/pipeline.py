from __future__ import annotations

import unicodedata
from pathlib import Path

from translation.config import TranslationConfig
from translation.models import Cue, PipelineResult, TranslationOutputPaths
from translation.subtitles import detect_subtitle_format, parse_subtitle_file


def run_translation_pipeline(subtitle_path: str | Path, config: TranslationConfig) -> PipelineResult:
    input_path = Path(subtitle_path)
    if not input_path.exists():
        raise FileNotFoundError(f"subtitle_path not found: {input_path}")

    input_format = detect_subtitle_format(input_path)
    output_paths = build_output_paths(input_path, config)
    _ensure_outputs_do_not_exist(output_paths, config.overwrite)

    cues = parse_subtitle_file(input_path)

    if config.dry_run:
        return PipelineResult(
            input_path=input_path,
            input_format=input_format,
            output_format="srt",
            output_paths=output_paths,
            dry_run=True,
            cue_count=len(cues),
            provider_called=False,
            first_cue_preview=_preview_cue(cues[0]),
            last_cue_preview=_preview_cue(cues[-1]),
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


def _preview_cue(cue: Cue) -> str:
    safe_source = "".join(
        character
        for character in cue.source
        if character.isspace() or not unicodedata.category(character).startswith("C")
    )
    preview = " ".join(safe_source.split())
    if len(preview) <= 80:
        return preview
    return f"{preview[:77]}..."
