from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from translation.batching import create_batches
from translation.config import TranslationConfig
from translation.models import Cue, PipelineResult, TranslationBatch, TranslationOutputPaths
from translation.prompts import build_translation_prompt
from translation.provider import OpenAICompatibleProvider, TranslationProvider
from translation.subtitles import (
    detect_subtitle_format,
    parse_subtitle_file,
    validate_translations,
    write_bilingual_srt,
    write_translated_srt,
)


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

    if not config.api_key:
        raise ValueError("TRANSLATION_API_KEY or --api-key is required for translation")

    provider = OpenAICompatibleProvider(config)
    batches = create_batches(cues, config.batch_size, config.context_before, config.context_after)
    all_translations: dict[str, str] = {}
    for batch in batches:
        batch_translations = _translate_batch_with_retries(provider, batch, config)
        all_translations.update(batch_translations)

    validate_translations(cues, all_translations)
    write_translated_srt(cues, all_translations, output_paths.translated_srt)
    write_bilingual_srt(cues, all_translations, output_paths.bilingual_srt)

    return PipelineResult(
        input_path=input_path,
        input_format=input_format,
        output_format="srt",
        output_paths=output_paths,
        dry_run=False,
        cue_count=len(cues),
        provider_called=True,
        first_cue_preview=_preview_cue(cues[0]),
        last_cue_preview=_preview_cue(cues[-1]),
    )


def parse_translation_response(
    response_text: str,
    expected_cues: list[Cue] | tuple[Cue, ...],
    batch_id: int,
) -> dict[str, str]:
    stripped = _strip_json_fence(response_text)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"batch_id {batch_id} translation response is not valid JSON") from exc

    if not isinstance(payload, list):
        raise ValueError(f"batch_id {batch_id} translation response must be a JSON array")
    if len(payload) != len(expected_cues):
        raise ValueError(
            f"batch_id {batch_id} translation count {len(payload)} does not match cue count {len(expected_cues)}"
        )

    translations: dict[str, str] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"batch_id {batch_id} translation item must be an object")
        cue_id = item.get("id")
        translation = item.get("translation")
        if not isinstance(cue_id, str):
            raise ValueError(f"batch_id {batch_id} translation item id must be a string")
        if not isinstance(translation, str):
            raise ValueError(f"batch_id {batch_id} cue id {cue_id} translation must be a string")
        translations[cue_id] = translation

    try:
        validate_translations(expected_cues, translations)
    except ValueError as exc:
        raise ValueError(f"batch_id {batch_id} {exc}") from exc
    return translations


def build_output_paths(input_path: Path, config: TranslationConfig) -> TranslationOutputPaths:
    output_dir = Path(config.output_dir) if config.output_dir else input_path.parent / "translated"
    return TranslationOutputPaths(
        output_dir=output_dir,
        translated_srt=output_dir / f"translated.{config.target_lang}.srt",
        bilingual_srt=output_dir / "bilingual.srt",
        translation_report=output_dir / "translation_report.md",
        global_context=output_dir / "global_context.md",
    )


def _translate_batch_with_retries(
    provider: TranslationProvider,
    batch: TranslationBatch,
    config: TranslationConfig,
) -> dict[str, str]:
    prompt = build_translation_prompt(batch, config.target_lang)
    attempts = _attempt_count(config)
    last_error: Exception | None = None
    for _attempt in range(1, attempts + 1):
        try:
            response_text = provider.translate_batch(prompt)
            return parse_translation_response(response_text, batch.cues, batch.batch_id)
        except (RuntimeError, ValueError) as exc:
            last_error = exc
    raise RuntimeError(f"batch_id {batch.batch_id} failed after {attempts} attempts") from last_error


def _attempt_count(config: TranslationConfig) -> int:
    return config.max_retries + 1


def _strip_json_fence(response_text: str) -> str:
    stripped = response_text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if lines[0] in {"```", "```json"} and lines[-1] == "```":
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _ensure_outputs_do_not_exist(paths: TranslationOutputPaths, overwrite: bool) -> None:
    if overwrite:
        return
    candidates = [paths.translated_srt, paths.bilingual_srt]
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
