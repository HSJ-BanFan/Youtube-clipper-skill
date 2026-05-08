#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from translation.config import load_config
from translation.models import PipelineResult
from translation.pipeline import run_translation_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate subtitles with B1-lite config/CLI pipeline.")
    parser.add_argument("subtitle_path", help="Input subtitle file (.srt or .vtt)")
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    parser.add_argument("--output-dir", help="Output directory; defaults to <subtitle_dir>/translated")
    parser.add_argument("--target-lang", dest="target_lang", help="Target language, e.g. zh-CN")
    parser.add_argument("--model", help="Translation model name")
    parser.add_argument("--review-model", dest="review_model", help="QA/review model name")
    parser.add_argument("--base-url", dest="base_url", help="OpenAI-compatible base URL")
    parser.add_argument("--glossary", dest="glossary_path", help="Glossary markdown path")
    parser.add_argument("--batch-size", dest="batch_size", type=int, help="Cue count per batch")
    parser.add_argument("--context-before", dest="context_before", type=int, help="Reference cue count before batch")
    parser.add_argument("--context-after", dest="context_after", type=int, help="Reference cue count after batch")
    parser.add_argument("--temperature", type=float, help="Model temperature")
    parser.add_argument("--max-retries", dest="max_retries", type=int, help="Maximum retries per batch")
    parser.add_argument("--cache-path", dest="cache_path", help="SQLite cache path")
    parser.add_argument("--qa", dest="qa_mode", choices=["suspicious-only", "none"], help="QA mode")
    parser.add_argument("--no-cache", dest="cache_enabled", action="store_false", default=None, help="Disable cache")
    parser.add_argument("--no-qa", dest="qa_mode", action="store_const", const="none", help="Disable QA")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and output paths only")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing outputs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cli_args = _namespace_to_config_args(args)

    try:
        config = load_config(cli_args=cli_args, env_path=args.env_file)
        result = run_translation_pipeline(args.subtitle_path, config)
    except (FileExistsError, FileNotFoundError, NotImplementedError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print_report_summary(result, config.to_safe_dict())
    return 0


def _namespace_to_config_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "base_url": args.base_url,
        "model": args.model,
        "review_model": args.review_model,
        "target_lang": args.target_lang,
        "batch_size": args.batch_size,
        "context_before": args.context_before,
        "context_after": args.context_after,
        "temperature": args.temperature,
        "max_retries": args.max_retries,
        "cache_enabled": args.cache_enabled,
        "cache_path": args.cache_path,
        "glossary_path": args.glossary_path,
        "qa_mode": args.qa_mode,
        "output_dir": args.output_dir,
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
    }


def _print_report_summary(result: PipelineResult, safe_config: dict[str, object]) -> None:
    print("Dry run" if result.dry_run else "Translation run")
    print(f"input_path: {result.input_path}")
    print(f"input_format: {result.input_format}")
    print(f"output_format: {result.output_format}")
    print(f"cue_count: {result.cue_count}")
    if result.first_cue_preview is not None:
        print(f"first_cue_preview: {result.first_cue_preview}")
    if result.last_cue_preview is not None:
        print(f"last_cue_preview: {result.last_cue_preview}")
    print("config:")
    for key, value in safe_config.items():
        print(f"  {key}: {value}")
    print("outputs:")
    print(f"  translated_srt: {result.output_paths.translated_srt}")
    print(f"  bilingual_srt: {result.output_paths.bilingual_srt}")
    print(f"  translation_report: {result.output_paths.translation_report}")
    print(f"  global_context: {result.output_paths.global_context}")


if __name__ == "__main__":
    raise SystemExit(main())
