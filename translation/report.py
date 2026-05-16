from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from translation.config import TranslationConfig
from translation.context import GlobalContext
from translation.glossary import Glossary
from translation.models import (
    SEGMENTATION_ARTIFACT_CUE_MAP,
    SEGMENTATION_ARTIFACT_REPORT,
    SEGMENTATION_ARTIFACT_SEGMENTED_SOURCE,
    SEGMENTATION_ARTIFACT_TRANSLATION_UNITS,
    MinimalBatchReportEntry,
    PipelineResult,
)
from translation.qa import QAIssue

REPORT_SCHEMA_VERSION = "translation-v2-report-v2"


@dataclass(frozen=True)
class AutoSubSegmentationStats:
    source_mode: str
    segmentation_strategy_version: str
    timing_strategy_version: str
    original_cue_count: int
    window_cue_count: int
    cleaned_active_token_count: int
    translation_unit_count: int
    translated_segment_unit_count: int
    skipped_padding_only_unit_count: int
    warning_count: int


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
    adaptive_concurrency_initial: int | None = None
    adaptive_concurrency_low_watermark: int | None = None
    adaptive_concurrency_high_watermark: int | None = None
    adaptive_concurrency_increase_events: int = 0
    adaptive_concurrency_decrease_events: int = 0
    adaptive_concurrency_pressure_events: int = 0
    batch_entries: list[MinimalBatchReportEntry] = field(default_factory=list)
    qa: QAStats | None = None
    auto_sub_segmentation: AutoSubSegmentationStats | None = None


def write_translation_report(
    path: Path,
    result: PipelineResult,
    config: TranslationConfig,
    stats: TranslationStats,
    glossary: Glossary,
    context: GlobalContext,
) -> None:
    safe_config = config.to_safe_dict()
    qa = stats.qa or QAStats(qa_mode=config.qa_mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_report(result, safe_config, stats, glossary, context, qa), encoding="utf-8")


def _safe_glossary_path(glossary: Glossary, safe_config: dict[str, Any]) -> str | Path | None:
    if glossary.path is not None:
        return glossary.path
    return safe_config["glossary_path"]


def _render_report(
    result: PipelineResult,
    safe_config: dict[str, Any],
    stats: TranslationStats,
    glossary: Glossary,
    context: GlobalContext,
    qa: QAStats,
) -> str:
    lines = ["# Translation Report", "", f"- report_schema_version: {REPORT_SCHEMA_VERSION}"]
    lines.extend(_render_section("Run Summary", _run_summary_entries(result, stats)))
    if stats.auto_sub_segmentation is not None:
        lines.extend(_render_section("Auto-sub Segmentation", _auto_sub_segmentation_entries(stats.auto_sub_segmentation)))
    lines.extend(_render_section("Config Snapshot", _config_snapshot_entries(safe_config, glossary, context)))
    lines.extend(_render_section("Cache Summary", _cache_summary_entries(safe_config, stats)))
    lines.extend(_render_section("Provider / Fallback Summary", _provider_summary_entries(safe_config, stats)))
    lines.extend(_render_section("Concurrency Summary", _concurrency_summary_entries(safe_config, stats)))
    lines.extend(_render_section("Shrink-Batch Summary", _shrink_summary_entries(stats.batch_entries)))
    lines.extend(_render_section("QA Summary", _qa_summary_entries(qa)))
    lines.extend(_render_section("QA", _qa_summary_entries(qa)))
    lines.extend(_render_section("QA Issues", _qa_issue_entries(qa)))
    lines.extend(_render_section("Batch Summary", _batch_summary_entries(stats.batch_entries)))
    batch_details = _batch_detail_entries(stats.batch_entries)
    lines.extend(_render_section("Batch Details", batch_details))
    lines.extend(_render_section("Batch Results", batch_details))
    lines.extend(_render_section("Terminal / Failure Summary", _terminal_summary_entries(stats, stats.batch_entries)))
    lines.extend(_render_section("Warnings", _warning_entries(stats, qa, stats.batch_entries)))
    lines.extend(_render_section("Output Artifacts", _output_artifact_entries(result, glossary)))
    lines.append("")
    return "\n".join(lines)


def _render_section(title: str, entries: Sequence[str]) -> list[str]:
    lines = ["", f"## {title}"]
    if entries:
        lines.extend(entries)
    else:
        lines.append("- none")
    return lines


def _run_summary_entries(result: PipelineResult, stats: TranslationStats) -> list[str]:
    final_status = "success" if stats.failed_batches == 0 else "completed_with_failures"
    return [
        f"- input_path: {result.input_path}",
        f"- input_format: {result.input_format}",
        f"- output_format: {result.output_format}",
        f"- cue_count: {result.cue_count}",
        f"- dry_run: {result.dry_run}",
        f"- provider_called: {result.provider_called}",
        f"- total_batches: {stats.total_batches}",
        f"- provider_calls: {stats.provider_calls}",
        f"- cache_hits: {stats.cache_hits}",
        f"- cache_misses: {stats.cache_misses}",
        f"- retries: {stats.retries}",
        f"- failed_batches: {stats.failed_batches}",
        f"- final_status: {final_status}",
    ]


def _auto_sub_segmentation_entries(stats: AutoSubSegmentationStats) -> list[str]:
    return [
        "- enabled: true",
        f"- source_mode: {stats.source_mode}",
        f"- segmentation_strategy_version: {stats.segmentation_strategy_version}",
        f"- timing_strategy_version: {stats.timing_strategy_version}",
        f"- original_cue_count: {stats.original_cue_count}",
        f"- window_cue_count: {stats.window_cue_count}",
        f"- cleaned_active_token_count: {stats.cleaned_active_token_count}",
        f"- translation_unit_count: {stats.translation_unit_count}",
        f"- translated_segment_unit_count: {stats.translated_segment_unit_count}",
        f"- skipped_padding_only_unit_count: {stats.skipped_padding_only_unit_count}",
        f"- warning_count: {stats.warning_count}",
        f"- segmentation_report: {SEGMENTATION_ARTIFACT_REPORT}",
        f"- translation_units: {SEGMENTATION_ARTIFACT_TRANSLATION_UNITS}",
        f"- cue_map: {SEGMENTATION_ARTIFACT_CUE_MAP}",
        f"- segmented_source: {SEGMENTATION_ARTIFACT_SEGMENTED_SOURCE}",
    ]


def _config_snapshot_entries(safe_config: dict[str, Any], glossary: Glossary, context: GlobalContext) -> list[str]:
    return [
        f"- target_lang: {safe_config['target_lang']}",
        f"- model: {safe_config['model']}",
        f"- review_model: {safe_config['review_model']}",
        f"- effective_review_model: {safe_config['effective_review_model']}",
        f"- base_url: {safe_config['base_url']}",
        f"- mode: {safe_config['mode']}",
        f"- batch_size: {safe_config['batch_size']}",
        f"- context_before: {safe_config['context_before']}",
        f"- context_after: {safe_config['context_after']}",
        f"- temperature: {safe_config['temperature']}",
        f"- max_retries: {safe_config['max_retries']}",
        f"- glossary_path: {_safe_glossary_path(glossary, safe_config)}",
        f"- qa_mode: {safe_config['qa_mode']}",
        f"- engine_version: {safe_config['engine_version']}",
        f"- structured_output: {safe_config['structured_output']}",
        f"- failure_mode: {safe_config['failure_mode']}",
        f"- main_model_alias: {safe_config['main_model_alias']}",
        f"- repair_model_alias: {safe_config['repair_model_alias']}",
        f"- fallback_model_alias: {safe_config['fallback_model_alias']}",
        f"- fallback_model: {safe_config['fallback_model']}",
        f"- batch_max_chars: {safe_config['batch_max_chars']}",
        f"- batch_max_cues: {safe_config['batch_max_cues']}",
        f"- output_schema_version: {safe_config['output_schema_version']}",
        f"- batching_strategy_version: {safe_config['batching_strategy_version']}",
        f"- glossary_hash: {glossary.hash}",
        f"- context_hash: {context.hash}",
    ]


def _cache_summary_entries(safe_config: dict[str, Any], stats: TranslationStats) -> list[str]:
    return [
        f"- cache_enabled: {safe_config['cache_enabled']}",
        f"- cache_path: {safe_config['cache_path']}",
        f"- cache_hits: {stats.cache_hits}",
        f"- cache_misses: {stats.cache_misses}",
        "- cache_scope: translation-stage only",
    ]


def _provider_summary_entries(safe_config: dict[str, Any], stats: TranslationStats) -> list[str]:
    return [
        f"- provider: {safe_config['provider']}",
        f"- model: {safe_config['model']}",
        f"- effective_review_model: {safe_config['effective_review_model']}",
        f"- provider_calls: {stats.provider_calls}",
        f"- fallback_provider_calls: {stats.fallback_provider_calls}",
        f"- main_model_alias: {safe_config['main_model_alias']}",
        f"- repair_model_alias: {safe_config['repair_model_alias']}",
        f"- fallback_model_alias: {safe_config['fallback_model_alias']}",
    ]


def _concurrency_summary_entries(safe_config: dict[str, Any], stats: TranslationStats) -> list[str]:
    return [
        f"- concurrency: {safe_config['concurrency']}",
        f"- adaptive_concurrency_enabled: {safe_config['adaptive_concurrency_enabled']}",
        f"- adaptive_concurrency_min: {safe_config['adaptive_concurrency_min']}",
        f"- adaptive_concurrency_max: {_render_optional_int(safe_config['adaptive_concurrency_max'])}",
        f"- adaptive_concurrency_initial: {_render_optional_int(stats.adaptive_concurrency_initial)}",
        f"- adaptive_concurrency_low_watermark: {_render_optional_int(stats.adaptive_concurrency_low_watermark)}",
        f"- adaptive_concurrency_high_watermark: {_render_optional_int(stats.adaptive_concurrency_high_watermark)}",
        f"- adaptive_concurrency_increase_events: {stats.adaptive_concurrency_increase_events}",
        f"- adaptive_concurrency_decrease_events: {stats.adaptive_concurrency_decrease_events}",
        f"- adaptive_concurrency_pressure_events: {stats.adaptive_concurrency_pressure_events}",
        f"- engine_version: {safe_config['engine_version']}",
        f"- structured_output: {safe_config['structured_output']}",
    ]


def _shrink_summary_entries(batch_entries: Sequence[MinimalBatchReportEntry]) -> list[str]:
    parent_batches_split = sum(1 for entry in batch_entries if entry.child_batch_ids or entry.split_reason is not None)
    child_batches_spawned = sum(len(entry.child_batch_ids) for entry in batch_entries)
    split_reasons = sorted({entry.split_reason for entry in batch_entries if entry.split_reason})
    split_attempts = [entry.split_attempt for entry in batch_entries if entry.split_attempt is not None]
    return [
        f"- parent_batches_split: {parent_batches_split}",
        f"- child_batches_spawned: {child_batches_spawned}",
        f"- split_reasons: {','.join(split_reasons) if split_reasons else 'none'}",
        f"- max_split_attempt: {max(split_attempts) if split_attempts else 'none'}",
    ]


def _qa_summary_entries(qa: QAStats) -> list[str]:
    return [
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
    ]


def _qa_issue_entries(qa: QAStats) -> list[str]:
    if qa.issues:
        return [f"- {issue.cue_id} | {issue.severity} | {issue.reason}" for issue in qa.issues]
    return ["- none"]


def _batch_summary_entries(batch_entries: Sequence[MinimalBatchReportEntry]) -> list[str]:
    fallback_final_routes = sum(1 for entry in batch_entries if entry.final_route_label == "fallback")
    main_final_routes = sum(1 for entry in batch_entries if entry.final_route_label == "main")
    cache_hit_batches = sum(1 for entry in batch_entries if entry.cache_hit)
    batches_with_retries = sum(1 for entry in batch_entries if entry.attempts > 1)
    return [
        f"- batch_entries: {len(batch_entries)}",
        f"- batches_with_fallback_final_route: {fallback_final_routes}",
        f"- batches_with_main_final_route: {main_final_routes}",
        f"- cache_hit_batches: {cache_hit_batches}",
        f"- batches_with_retries: {batches_with_retries}",
    ]


def _batch_detail_entries(batch_entries: Sequence[MinimalBatchReportEntry]) -> list[str]:
    if not batch_entries:
        return ["- none"]
    return [_render_batch_entry(entry) for entry in batch_entries]


def _terminal_summary_entries(
    stats: TranslationStats,
    batch_entries: Sequence[MinimalBatchReportEntry],
) -> list[str]:
    final_status = "success" if stats.failed_batches == 0 else "completed_with_failures"
    terminal_batch_ids = [str(entry.batch_id) for entry in batch_entries if entry.state.value != "success"]
    return [
        f"- final_status: {final_status}",
        f"- failed_batches: {stats.failed_batches}",
        f"- terminal_batch_ids: {','.join(terminal_batch_ids) if terminal_batch_ids else 'none'}",
    ]


def _render_optional_int(value: int | None) -> str:
    return "none" if value is None else str(value)


def _warning_entries(
    stats: TranslationStats,
    qa: QAStats,
    batch_entries: Sequence[MinimalBatchReportEntry],
) -> list[str]:
    warnings: list[str] = []
    if qa.qa_failed > 0:
        warnings.append("- QA failures occurred but translation-stage output was preserved.")
    if stats.fallback_provider_calls > 0 or any(entry.final_route_label == "fallback" for entry in batch_entries):
        warnings.append("- Fallback route used for one or more batches.")
    if any(entry.child_batch_ids or entry.split_reason is not None for entry in batch_entries):
        warnings.append("- Shrink-batch compensation activated for one or more parent batches.")
    if stats.failed_batches > 0:
        warnings.append("- One or more batches terminated with failure metadata in report.")
    return warnings or ["- none"]


def _output_artifact_entries(result: PipelineResult, glossary: Glossary) -> list[str]:
    return [
        f"- output_dir: {result.output_paths.output_dir}",
        f"- translated_srt: {result.output_paths.translated_srt}",
        f"- bilingual_srt: {result.output_paths.bilingual_srt}",
        f"- translation_report: {result.output_paths.translation_report}",
        f"- global_context: {result.output_paths.global_context}",
        f"- glossary_truncated: {glossary.truncated}",
    ]


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
