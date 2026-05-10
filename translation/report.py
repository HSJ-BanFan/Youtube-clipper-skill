from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from translation.config import TranslationConfig
from translation.context import GlobalContext
from translation.glossary import Glossary
from translation.models import PipelineResult
from translation.qa import QAIssue


@dataclass
class QAStats:
    qa_mode: str = "none"
    qa_candidates: int = 0
    qa_provider_calls: int = 0
    qa_fixed: int = 0
    qa_kept: int = 0
    qa_failed: int = 0
    qa_prompt_version: str = ""
    issues: tuple[QAIssue, ...] = ()


@dataclass
class TranslationStats:
    total_batches: int = 0
    provider_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    retries: int = 0
    failed_batches: int = 0
    qa: QAStats | None = None


def write_translation_report(
    path: Path,
    result: PipelineResult,
    config: TranslationConfig,
    stats: TranslationStats,
    glossary: Glossary,
    context: GlobalContext,
) -> None:
    safe_config = config.to_safe_dict()
    entries = {
        "input_path": result.input_path,
        "input_format": result.input_format,
        "output_format": result.output_format,
        "cue_count": result.cue_count,
        "target_lang": safe_config["target_lang"],
        "model": safe_config["model"],
        "provider": safe_config["provider"],
        "base_url": safe_config["base_url"],
        "batch_size": safe_config["batch_size"],
        "context_before": safe_config["context_before"],
        "context_after": safe_config["context_after"],
        "cache_enabled": safe_config["cache_enabled"],
        "cache_path": safe_config["cache_path"],
        "glossary_path": _safe_glossary_path(glossary, safe_config),
        "glossary_hash": glossary.hash,
        "context_hash": context.hash,
        "total_batches": stats.total_batches,
        "provider_calls": stats.provider_calls,
        "cache_hits": stats.cache_hits,
        "cache_misses": stats.cache_misses,
        "retries": stats.retries,
        "failed_batches": stats.failed_batches,
        "output_dir": result.output_paths.output_dir,
        "translated_srt": result.output_paths.translated_srt,
        "bilingual_srt": result.output_paths.bilingual_srt,
        "translation_report": result.output_paths.translation_report,
        "global_context": result.output_paths.global_context,
        "glossary_truncated": glossary.truncated,
    }
    qa = stats.qa or QAStats(qa_mode=config.qa_mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_report(entries, qa), encoding="utf-8")


def _safe_glossary_path(glossary: Glossary, safe_config: dict[str, Any]) -> str | Path | None:
    if glossary.path is not None:
        return glossary.path
    return safe_config["glossary_path"]


def _render_report(entries: dict[str, Any], qa: QAStats) -> str:
    lines = ["# Translation Report", ""]
    lines.extend(f"- {key}: {value}" for key, value in entries.items())
    lines.extend(
        [
            "",
            "## QA",
            f"- qa_mode: {qa.qa_mode}",
            f"- qa_candidates: {qa.qa_candidates}",
            f"- qa_provider_calls: {qa.qa_provider_calls}",
            f"- qa_fixed: {qa.qa_fixed}",
            f"- qa_kept: {qa.qa_kept}",
            f"- qa_failed: {qa.qa_failed}",
            f"- qa_prompt_version: {qa.qa_prompt_version}",
            "",
            "## QA Issues",
        ]
    )
    if qa.issues:
        lines.extend(f"- {issue.cue_id} | {issue.severity} | {issue.reason}" for issue in qa.issues)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)
