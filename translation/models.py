from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


# V1 Models - Current runtime models (unchanged)


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
class TranslationBatch:
    batch_id: int
    cues: tuple[Cue, ...]
    context_before: tuple[Cue, ...]
    context_after: tuple[Cue, ...]


SEGMENTATION_ARTIFACT_SEGMENTED_SOURCE = "segmented_source.srt"
SEGMENTATION_ARTIFACT_TRANSLATION_UNITS = "translation_units.json"
SEGMENTATION_ARTIFACT_CUE_MAP = "cue_map.json"
SEGMENTATION_ARTIFACT_REPORT = "segmentation_report.md"


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


# V2 Models - Passive records for future parser/report work


class BatchState(str, Enum):
    """Batch processing state for passive scaffolding."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    RETRYING = "retrying"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_PERMANENT = "failed_permanent"
    SUSPICIOUS = "suspicious"
    REPAIRING = "repairing"
    REPAIRED = "repaired"


class FailureMode(str, Enum):
    """Batch failure classification for passive scaffolding."""

    STRICT = "strict"
    PARTIAL = "partial"
    INTERACTIVE = "interactive"


class ErrorType(str, Enum):
    """Error type classification for parser/report validation."""

    INVALID_JSON = "invalid_json"
    SCHEMA_MISMATCH = "schema_mismatch"
    MISSING_REQUIRED_CUE_ID = "missing_required_cue_id"
    DUPLICATE_CUE_ID = "duplicate_cue_id"
    INVALID_CUE_ID = "invalid_cue_id"
    CONTEXT_CUE_OUTPUT_VIOLATION = "context_cue_output_violation"
    EMPTY_TRANSLATION = "empty_translation"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_HTTP_5XX = "provider_http_5xx"
    PROVIDER_REQUEST_FAILED = "provider_request_failed"
    PROVIDER_MISSING_CHOICES = "provider_missing_choices"


@dataclass(frozen=True)
class CueRecord:
    """V2 cue record - passive scaffolding for future parser work."""

    cue_id: str
    original_index: int
    start: str
    end: str
    source_text: str
    raw_timing: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class AttemptRecord:
    """V2 attempt record for retry tracking."""

    attempt_index: int
    batch_id: int
    model_alias: str
    error_type: ErrorType | None
    duration_ms: int
    cache_hit: bool
    result_state: BatchState
    route_label: str | None = None


@dataclass(frozen=True)
class BatchRecord:
    """V2 batch record with hierarchical batch tracking."""

    batch_id: int
    target_cues: tuple[CueRecord, ...]
    context_before: tuple[CueRecord, ...]
    context_after: tuple[CueRecord, ...]
    status: BatchState
    parent_batch_id: int | None = None
    child_batch_ids: tuple[int, ...] = ()
    split_reason: str | None = None
    split_attempt: int | None = None
    split_strategy_version: str | None = None
    original_target_cue_range: tuple[int, int] | None = None


@dataclass(frozen=True)
class MinimalBatchReportEntry:
    """Lightweight batch report entry for summary reporting."""

    batch_id: int
    state: BatchState
    cue_count: int
    attempts: int
    cache_hit: bool
    cue_range: tuple[int, int] | None = None
    attempt: int | None = None
    error_type: ErrorType | None = None
    duration_ms: int | None = None
    failure_mode: FailureMode | None = None
    error_summary: str | None = None
    final_route_label: str | None = None
    parent_batch_id: int | None = None
    child_batch_ids: tuple[int, ...] = ()
    split_reason: str | None = None
    split_attempt: int | None = None
    split_strategy_version: str | None = None
    original_target_cue_range: tuple[int, int] | None = None
