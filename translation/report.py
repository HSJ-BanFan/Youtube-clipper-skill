from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from translation.config import TranslationConfig
from translation.context import GlobalContext
from translation.glossary import Glossary
from translation.models import MinimalBatchReportEntry, PipelineResult
from translation.qa import QAIssue


@dataclass
class QAStats:
    qa_mode: str = "none"
    qa_candidates: int = 0
    qa_reviewed: int = 0
    qa_provider_calls: int = 0
    qa_fixed: int = 0
    qa_kept: int = 0
    qa_failed: int = 0
    qa_parser_failures: int = 0
    qa_provider_failures: int = 0
    qa_skipped: int = 0
    qa_prompt_version: str = ""
    issues: tuple[QAIssue, ...] = ()


@dataclass
class TranslationStats:
    total_batches: int = 0
    provider_calls: int = 0
    fallback_provider_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    retries: int = 0
    failed_batches: int = 0
    batch_entries: list[MinimalBatchReportEntry] = field(default_factory=list)
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
        "engine_version": safe_config["engine_version"],
        "structured_output": safe_config["structured_output"],
        "failure_mode": safe_config["failure_mode"],
        "main_model_alias": safe_config["main_model_alias"],
        "repair_model_alias": safe_config["repair_model_alias"],
        "fallback_model_alias": safe_config["fallback_model_alias"],
        "batch_max_chars": safe_config["batch_max_chars"],
        "batch_max_cues": safe_config["batch_max_cues"],
        "output_schema_version": safe_config["output_schema_version"],
        "batching_strategy_version": safe_config["batching_strategy_version"],
        "glossary_hash": glossary.hash,
        "context_hash": context.hash,
        "total_batches": stats.total_batches,
        "provider_calls": stats.provider_calls,
        "fallback_provider_calls": stats.fallback_provider_calls,
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
    path.write_text(_render_report(entries, qa, stats.batch_entries), encoding="utf-8")


def _safe_glossary_path(glossary: Glossary, safe_config: dict[str, Any]) -> str | Path | None:
    if glossary.path is not None:
        return glossary.path
    return safe_config["glossary_path"]


def _render_report(entries: dict[str, Any], qa: QAStats, batch_entries: Sequence[MinimalBatchReportEntry]) -> str:
    lines = ["# Translation Report", ""]
    lines.extend(f"- {key}: {value}" for key, value in entries.items())
    lines.extend(
        [
            "",
            "## QA",
            f"- qa_mode: {qa.qa_mode}",
            f"- qa_candidates: {qa.qa_candidates}",
            f"- qa_reviewed: {qa.qa_reviewed}",
            f"- qa_provider_calls: {qa.qa_provider_calls}",
            f"- qa_fixed: {qa.qa_fixed}",
            f"- qa_kept: {qa.qa_kept}",
            f"- qa_failed: {qa.qa_failed}",
            f"- qa_parser_failures: {qa.qa_parser_failures}",
            f"- qa_provider_failures: {qa.qa_provider_failures}",
            f"- qa_skipped: {qa.qa_skipped}",
            f"- qa_prompt_version: {qa.qa_prompt_version}",
            "",
            "## QA Issues",
        ]
    )
    if qa.issues:
        lines.extend(f"- {issue.cue_id} | {issue.severity} | {issue.reason}" for issue in qa.issues)
    else:
        lines.append("- none")
    if batch_entries:
        lines.extend(["", "## Batch Results"])
        lines.extend(_render_batch_entry(entry) for entry in batch_entries)
    lines.append("")
    return "\n".join(lines)



def _render_batch_entry(entry: MinimalBatchReportEntry) -> str:
    cue_range = "none" if entry.cue_range is None else f"{entry.cue_range[0]}-{entry.cue_range[1]}"
    attempt = entry.attempt if entry.attempt is not None else entry.attempts
    error_type = entry.error_type.value if entry.error_type is not None else "none"
    duration_ms = "none" if entry.duration_ms is None else str(entry.duration_ms)
    final_route_label = entry.final_route_label or "none"
    parent_batch_id = "none" if entry.parent_batch_id is None else str(entry.parent_batch_id)
    child_batch_ids = ",".join(str(batch_id) for batch_id in entry.child_batch_ids) if entry.child_batch_ids else "none"
    split_reason = entry.split_reason or "none"
    split_attempt = "none" if entry.split_attempt is None else str(entry.split_attempt)
    split_strategy_version = entry.split_strategy_version or "none"
    original_target_cue_range = (
        "none"
        if entry.original_target_cue_range is None
        else f"{entry.original_target_cue_range[0]}-{entry.original_target_cue_range[1]}"
    )
    return (
        f"- batch_id: {entry.batch_id} | cue_range: {cue_range} | status: {entry.state.value} "
        f"| attempt: {attempt} | error_type: {error_type} | cache_hit: {entry.cache_hit} | duration_ms: {duration_ms}"
        f" | final_route_label: {final_route_label} | parent_batch_id: {parent_batch_id}"
        f" | child_batch_ids: {child_batch_ids} | split_reason: {split_reason}"
        f" | split_attempt: {split_attempt} | split_strategy_version: {split_strategy_version}"
        f" | original_target_cue_range: {original_target_cue_range}"
    )
