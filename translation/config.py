from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


VALID_MODES = {"fast", "balanced", "publish"}
VALID_ENGINE_VERSIONS = {"v1", "v2"}
VALID_FAILURE_MODES = {"strict", "partial", "interactive"}
VALID_OUTPUT_SCHEMA_VERSIONS = {"v1"}
VALID_BATCHING_STRATEGY_VERSIONS = {"v1"}
VALID_AUTO_SUB_SOURCE_MODES = {"single_file", "full_vtt_window"}
VALID_SEMANTIC_SEGMENTATION_MODES = {"hybrid", "llm"}
TARGET_LANG_PATTERN = re.compile(r"^[A-Za-z0-9-]+$")


ENV_MAPPING = {
    "provider": "TRANSLATION_PROVIDER",
    "base_url": "TRANSLATION_BASE_URL",
    "api_key": "TRANSLATION_API_KEY",
    "model": "TRANSLATION_MODEL",
    "review_model": "TRANSLATION_REVIEW_MODEL",
    "target_lang": "TRANSLATION_TARGET_LANG",
    "mode": "TRANSLATION_MODE",
    "batch_size": "TRANSLATION_BATCH_SIZE",
    "context_before": "TRANSLATION_CONTEXT_BEFORE",
    "context_after": "TRANSLATION_CONTEXT_AFTER",
    "temperature": "TRANSLATION_TEMPERATURE",
    "max_retries": "TRANSLATION_MAX_RETRIES",
    "cache_enabled": "TRANSLATION_CACHE",
    "cache_path": "TRANSLATION_CACHE_PATH",
    "glossary_path": "TRANSLATION_GLOSSARY_PATH",
    "qa_mode": "TRANSLATION_QA",
    "engine_version": "TRANSLATION_ENGINE_VERSION",
    "structured_output": "TRANSLATION_STRUCTURED_OUTPUT",
    "failure_mode": "TRANSLATION_FAILURE_MODE",
    "main_model_alias": "TRANSLATION_MAIN_MODEL_ALIAS",
    "repair_model_alias": "TRANSLATION_REPAIR_MODEL_ALIAS",
    "fallback_model_alias": "TRANSLATION_FALLBACK_MODEL_ALIAS",
    "fallback_model": "TRANSLATION_FALLBACK_MODEL",
    "batch_max_chars": "TRANSLATION_BATCH_MAX_CHARS",
    "batch_max_cues": "TRANSLATION_BATCH_MAX_CUES",
    "concurrency": "TRANSLATION_CONCURRENCY",
    "adaptive_concurrency_enabled": "TRANSLATION_ADAPTIVE_CONCURRENCY_ENABLED",
    "adaptive_concurrency_min": "TRANSLATION_ADAPTIVE_CONCURRENCY_MIN",
    "adaptive_concurrency_max": "TRANSLATION_ADAPTIVE_CONCURRENCY_MAX",
    "output_schema_version": "TRANSLATION_OUTPUT_SCHEMA_VERSION",
    "batching_strategy_version": "TRANSLATION_BATCHING_STRATEGY_VERSION",
    "preprocess_auto_subs": "TRANSLATION_PREPROCESS_AUTO_SUBS",
    "auto_sub_source_mode": "TRANSLATION_AUTO_SUB_SOURCE_MODE",
    "auto_sub_full_vtt_path": "TRANSLATION_AUTO_SUB_FULL_VTT_PATH",
    "auto_sub_clip_start_ms": "TRANSLATION_AUTO_SUB_CLIP_START_MS",
    "auto_sub_clip_end_ms": "TRANSLATION_AUTO_SUB_CLIP_END_MS",
    "auto_sub_padding_before_ms": "TRANSLATION_AUTO_SUB_PADDING_BEFORE_MS",
    "auto_sub_padding_after_ms": "TRANSLATION_AUTO_SUB_PADDING_AFTER_MS",
    "segment_max_unit_chars": "TRANSLATION_SEGMENT_MAX_UNIT_CHARS",
    "segment_max_unit_duration_ms": "TRANSLATION_SEGMENT_MAX_UNIT_DURATION_MS",
    "segment_max_source_cues": "TRANSLATION_SEGMENT_MAX_SOURCE_CUES",
    "segment_max_sentences": "TRANSLATION_SEGMENT_MAX_SENTENCES",
    "semantic_segmentation_enabled": "TRANSLATION_SEMANTIC_SEGMENTATION",
    "semantic_segmentation_mode": "TRANSLATION_SEMANTIC_SEGMENTATION_MODE",
    "semantic_segmentation_model": "TRANSLATION_SEMANTIC_SEGMENTATION_MODEL",
    "semantic_segmentation_prompt_version": "TRANSLATION_SEMANTIC_SEGMENTATION_PROMPT_VERSION",
    "semantic_segmentation_max_unit_chars": "TRANSLATION_SEMANTIC_SEGMENTATION_MAX_UNIT_CHARS",
    "semantic_segmentation_max_unit_duration_ms": "TRANSLATION_SEMANTIC_SEGMENTATION_MAX_UNIT_DURATION_MS",
    "semantic_segmentation_min_unit_duration_ms": "TRANSLATION_SEMANTIC_SEGMENTATION_MIN_UNIT_DURATION_MS",
    "semantic_segmentation_max_tokens_per_request": "TRANSLATION_SEMANTIC_SEGMENTATION_MAX_TOKENS_PER_REQUEST",
    "semantic_segmentation_fallback_to_rules": "TRANSLATION_SEMANTIC_SEGMENTATION_FALLBACK_TO_RULES",
}


@dataclass(frozen=True)
class TranslationConfig:
    provider: str = "openai-compatible"
    base_url: str = "http://127.0.0.1:8317/v1"
    api_key: str | None = field(default=None, repr=False)
    model: str = "deepseek-chat"
    review_model: str | None = None
    target_lang: str = "zh-CN"
    mode: str = "balanced"
    batch_size: int = 80
    context_before: int = 10
    context_after: int = 10
    temperature: float = 0.1
    max_retries: int = 3
    cache_enabled: bool = True
    cache_path: str = ".translation_cache.sqlite3"
    glossary_path: str = "glossary.md"
    qa_mode: str = "suspicious-only"
    engine_version: str = "v1"
    structured_output: bool = False
    failure_mode: str = "strict"
    main_model_alias: str = "main"
    repair_model_alias: str = "repair"
    fallback_model_alias: str = "fallback"
    fallback_model: str | None = None
    batch_max_chars: int | None = None
    batch_max_cues: int | None = None
    concurrency: int = 1
    adaptive_concurrency_enabled: bool = False
    adaptive_concurrency_min: int = 1
    adaptive_concurrency_max: int | None = None
    output_schema_version: str = "v1"
    batching_strategy_version: str = "v1"
    preprocess_auto_subs: bool = False
    auto_sub_source_mode: str = "single_file"
    auto_sub_full_vtt_path: str | None = None
    auto_sub_clip_start_ms: int | None = None
    auto_sub_clip_end_ms: int | None = None
    auto_sub_padding_before_ms: int = 10_000
    auto_sub_padding_after_ms: int = 10_000
    segment_max_unit_chars: int = 180
    segment_max_unit_duration_ms: int = 7_000
    segment_max_source_cues: int = 5
    segment_max_sentences: int = 2
    semantic_segmentation_enabled: bool = False
    semantic_segmentation_mode: str = "hybrid"
    semantic_segmentation_model: str | None = None
    semantic_segmentation_prompt_version: str = "cycle3a-semantic-v1"
    semantic_segmentation_max_unit_chars: int = 220
    semantic_segmentation_max_unit_duration_ms: int = 9_000
    semantic_segmentation_min_unit_duration_ms: int = 800
    semantic_segmentation_max_tokens_per_request: int = 350
    semantic_segmentation_fallback_to_rules: bool = True
    output_dir: str | None = None
    output_path: str | None = None
    dry_run: bool = False
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.provider != "openai-compatible":
            raise ValueError("only openai-compatible is supported in B1-lite")
        object.__setattr__(self, "mode", self.mode.strip().lower())
        object.__setattr__(self, "qa_mode", self.qa_mode.strip().lower())
        object.__setattr__(self, "engine_version", self.engine_version.strip().lower())
        object.__setattr__(self, "failure_mode", self.failure_mode.strip().lower())
        object.__setattr__(self, "semantic_segmentation_mode", self.semantic_segmentation_mode.strip().lower())
        if self.qa_mode == "off":
            object.__setattr__(self, "qa_mode", "none")
        if self.mode not in VALID_MODES:
            valid_modes = ", ".join(sorted(VALID_MODES))
            raise ValueError(f"mode must be one of: {valid_modes}")
        if self.engine_version not in VALID_ENGINE_VERSIONS:
            valid_engine_versions = ", ".join(sorted(VALID_ENGINE_VERSIONS))
            raise ValueError(f"engine_version must be one of: {valid_engine_versions}")
        if not isinstance(self.structured_output, bool):
            raise ValueError("structured_output must be a boolean")
        if not TARGET_LANG_PATTERN.fullmatch(self.target_lang):
            raise ValueError("target_lang must contain only letters, numbers, and hyphens")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if self.context_before < 0:
            raise ValueError("context_before must be greater than or equal to 0")
        if self.context_after < 0:
            raise ValueError("context_after must be greater than or equal to 0")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to 0")
        if self.qa_mode not in {"suspicious-only", "none"}:
            raise ValueError("qa_mode must be suspicious-only, none, or off (alias for none)")
        if self.failure_mode not in VALID_FAILURE_MODES:
            valid_failure_modes = ", ".join(sorted(VALID_FAILURE_MODES))
            raise ValueError(f"failure_mode must be one of: {valid_failure_modes}")
        if self.batch_max_chars is not None and self.batch_max_chars <= 0:
            raise ValueError("batch_max_chars must be greater than 0 when set")
        if self.batch_max_cues is not None and self.batch_max_cues <= 0:
            raise ValueError("batch_max_cues must be greater than 0 when set")
        if self.concurrency <= 0:
            raise ValueError("concurrency must be greater than 0")
        if not isinstance(self.adaptive_concurrency_enabled, bool):
            raise ValueError("adaptive_concurrency_enabled must be a boolean")
        if self.adaptive_concurrency_min <= 0:
            raise ValueError("adaptive_concurrency_min must be greater than 0")
        if self.adaptive_concurrency_max is not None and self.adaptive_concurrency_max < self.adaptive_concurrency_min:
            raise ValueError("adaptive_concurrency_max must be greater than or equal to adaptive_concurrency_min")
        if self.output_schema_version not in VALID_OUTPUT_SCHEMA_VERSIONS:
            valid_output_schema_versions = ", ".join(sorted(VALID_OUTPUT_SCHEMA_VERSIONS))
            raise ValueError(f"output_schema_version must be one of: {valid_output_schema_versions}")
        if self.batching_strategy_version not in VALID_BATCHING_STRATEGY_VERSIONS:
            valid_batching_strategy_versions = ", ".join(sorted(VALID_BATCHING_STRATEGY_VERSIONS))
            raise ValueError(f"batching_strategy_version must be one of: {valid_batching_strategy_versions}")
        if self.auto_sub_source_mode not in VALID_AUTO_SUB_SOURCE_MODES:
            valid_auto_sub_source_modes = ", ".join(sorted(VALID_AUTO_SUB_SOURCE_MODES))
            raise ValueError(f"auto_sub_source_mode must be one of: {valid_auto_sub_source_modes}")
        if not isinstance(self.preprocess_auto_subs, bool):
            raise ValueError("preprocess_auto_subs must be a boolean")
        if self.auto_sub_padding_before_ms < 0:
            raise ValueError("auto_sub_padding_before_ms must be greater than or equal to 0")
        if self.auto_sub_padding_after_ms < 0:
            raise ValueError("auto_sub_padding_after_ms must be greater than or equal to 0")
        if self.segment_max_unit_chars <= 0:
            raise ValueError("segment_max_unit_chars must be greater than 0")
        if self.segment_max_unit_duration_ms <= 0:
            raise ValueError("segment_max_unit_duration_ms must be greater than 0")
        if self.segment_max_source_cues <= 0:
            raise ValueError("segment_max_source_cues must be greater than 0")
        if self.segment_max_sentences <= 0:
            raise ValueError("segment_max_sentences must be greater than 0")
        if (
            self.auto_sub_clip_start_ms is not None
            and self.auto_sub_clip_end_ms is not None
            and self.auto_sub_clip_start_ms >= self.auto_sub_clip_end_ms
        ):
            raise ValueError("auto_sub_clip_start_ms must be before auto_sub_clip_end_ms")
        if self.preprocess_auto_subs and self.auto_sub_source_mode == "full_vtt_window":
            if not self.auto_sub_full_vtt_path:
                raise ValueError("auto_sub_full_vtt_path is required for full_vtt_window")
            if self.auto_sub_clip_start_ms is None:
                raise ValueError("auto_sub_clip_start_ms is required for full_vtt_window")
            if self.auto_sub_clip_end_ms is None:
                raise ValueError("auto_sub_clip_end_ms is required for full_vtt_window")
        if self.output_dir and self.output_path:
            raise ValueError("output_dir and output_path cannot both be set")
        if not isinstance(self.semantic_segmentation_enabled, bool):
            raise ValueError("semantic_segmentation_enabled must be a boolean")
        if self.semantic_segmentation_enabled:
            if not self.preprocess_auto_subs:
                raise ValueError("semantic_segmentation_enabled requires preprocess_auto_subs=True")
            if self.semantic_segmentation_mode not in VALID_SEMANTIC_SEGMENTATION_MODES:
                valid = ", ".join(sorted(VALID_SEMANTIC_SEGMENTATION_MODES))
                raise ValueError(f"semantic_segmentation_mode must be one of: {valid}")
        if self.semantic_segmentation_max_unit_chars <= 0:
            raise ValueError("semantic_segmentation_max_unit_chars must be greater than 0")
        if self.semantic_segmentation_max_unit_duration_ms <= 0:
            raise ValueError("semantic_segmentation_max_unit_duration_ms must be greater than 0")
        if self.semantic_segmentation_min_unit_duration_ms <= 0:
            raise ValueError("semantic_segmentation_min_unit_duration_ms must be greater than 0")
        if self.semantic_segmentation_max_tokens_per_request <= 0:
            raise ValueError("semantic_segmentation_max_tokens_per_request must be greater than 0")

    @property
    def effective_review_model(self) -> str:
        return self.review_model or self.model

    @property
    def effective_adaptive_concurrency_max(self) -> int:
        configured_max = self.concurrency if self.adaptive_concurrency_max is None else self.adaptive_concurrency_max
        return min(self.concurrency, configured_max)

    @property
    def effective_semantic_segmentation_model(self) -> str:
        return self.semantic_segmentation_model or self.model

    @property
    def effective_semantic_segmentation_mode(self) -> str:
        return self.semantic_segmentation_mode if self.semantic_segmentation_enabled else "off"

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "base_url": _redact_url(self.base_url),
            "model": self.model,
            "review_model": self.review_model,
            "effective_review_model": self.effective_review_model,
            "target_lang": self.target_lang,
            "mode": self.mode,
            "batch_size": self.batch_size,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "cache_enabled": self.cache_enabled,
            "cache_path": self.cache_path,
            "glossary_path": self.glossary_path,
            "qa_mode": self.qa_mode,
            "engine_version": self.engine_version,
            "structured_output": self.structured_output,
            "failure_mode": self.failure_mode,
            "main_model_alias": self.main_model_alias,
            "repair_model_alias": self.repair_model_alias,
            "fallback_model_alias": self.fallback_model_alias,
            "fallback_model": self.fallback_model,
            "batch_max_chars": self.batch_max_chars,
            "batch_max_cues": self.batch_max_cues,
            "concurrency": self.concurrency,
            "adaptive_concurrency_enabled": self.adaptive_concurrency_enabled,
            "adaptive_concurrency_min": self.adaptive_concurrency_min,
            "adaptive_concurrency_max": self.adaptive_concurrency_max,
            "output_schema_version": self.output_schema_version,
            "batching_strategy_version": self.batching_strategy_version,
            "preprocess_auto_subs": self.preprocess_auto_subs,
            "auto_sub_source_mode": self.auto_sub_source_mode,
            "auto_sub_full_vtt_path": self.auto_sub_full_vtt_path,
            "auto_sub_clip_start_ms": self.auto_sub_clip_start_ms,
            "auto_sub_clip_end_ms": self.auto_sub_clip_end_ms,
            "auto_sub_padding_before_ms": self.auto_sub_padding_before_ms,
            "auto_sub_padding_after_ms": self.auto_sub_padding_after_ms,
            "segment_max_unit_chars": self.segment_max_unit_chars,
            "segment_max_unit_duration_ms": self.segment_max_unit_duration_ms,
            "segment_max_source_cues": self.segment_max_source_cues,
            "segment_max_sentences": self.segment_max_sentences,
            "semantic_segmentation_enabled": self.semantic_segmentation_enabled,
            "semantic_segmentation_mode": self.semantic_segmentation_mode,
            "effective_semantic_segmentation_mode": self.effective_semantic_segmentation_mode,
            "semantic_segmentation_model": self.semantic_segmentation_model,
            "effective_semantic_segmentation_model": self.effective_semantic_segmentation_model,
            "semantic_segmentation_prompt_version": self.semantic_segmentation_prompt_version,
            "semantic_segmentation_max_unit_chars": self.semantic_segmentation_max_unit_chars,
            "semantic_segmentation_max_unit_duration_ms": self.semantic_segmentation_max_unit_duration_ms,
            "semantic_segmentation_min_unit_duration_ms": self.semantic_segmentation_min_unit_duration_ms,
            "semantic_segmentation_max_tokens_per_request": self.semantic_segmentation_max_tokens_per_request,
            "semantic_segmentation_fallback_to_rules": self.semantic_segmentation_fallback_to_rules,
            "output_dir": self.output_dir,
            "output_path": self.output_path,
            "dry_run": self.dry_run,
            "overwrite": self.overwrite,
        }


def load_config(
    cli_args: Mapping[str, Any] | None = None,
    env_path: str | Path | None = ".env",
    environ: Mapping[str, str] | None = None,
) -> TranslationConfig:
    env = dict(os.environ if environ is None else environ)
    if env_path is not None:
        for key, value in _load_env_file(Path(env_path)).items():
            env.setdefault(key, value)

    values: dict[str, Any] = {}
    for field_name, env_name in ENV_MAPPING.items():
        raw_value = env.get(env_name)
        if raw_value is not None and raw_value != "":
            values[field_name] = _coerce_value(field_name, raw_value)

    for field_name, value in (cli_args or {}).items():
        if value is not None:
            values[field_name] = value

    return TranslationConfig(**values)


def _load_env_file(env_path: Path) -> dict[str, str]:
    if not env_path.exists():
        return {}

    try:
        from dotenv import dotenv_values
    except ModuleNotFoundError:
        return _parse_env_file(env_path)

    return {
        key: value
        for key, value in dotenv_values(env_path).items()
        if key and value is not None
    }


def _parse_env_file(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _coerce_value(field_name: str, raw_value: str) -> Any:
    try:
        if field_name in {
            "batch_size",
            "context_before",
            "context_after",
            "max_retries",
            "batch_max_chars",
            "batch_max_cues",
            "concurrency",
            "adaptive_concurrency_min",
            "adaptive_concurrency_max",
            "auto_sub_clip_start_ms",
            "auto_sub_clip_end_ms",
            "auto_sub_padding_before_ms",
            "auto_sub_padding_after_ms",
            "segment_max_unit_chars",
            "segment_max_unit_duration_ms",
            "segment_max_source_cues",
            "segment_max_sentences",
            "semantic_segmentation_max_unit_chars",
            "semantic_segmentation_max_unit_duration_ms",
            "semantic_segmentation_min_unit_duration_ms",
            "semantic_segmentation_max_tokens_per_request",
        }:
            return int(raw_value)
        if field_name == "temperature":
            return float(raw_value)
        if field_name in {"cache_enabled", "structured_output", "adaptive_concurrency_enabled", "preprocess_auto_subs", "semantic_segmentation_enabled", "semantic_segmentation_fallback_to_rules"}:
            return _parse_bool(raw_value)
    except ValueError as exc:
        env_name = ENV_MAPPING.get(field_name, field_name)
        raise ValueError(f"invalid value for {env_name}: {raw_value!r}") from exc
    return raw_value


def _parse_bool(raw_value: str) -> bool:
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {raw_value}")


def _redact_url(url: str) -> str:
    if "@" not in url:
        return url
    scheme, rest = url.split("://", 1) if "://" in url else ("", url)
    host = rest.rsplit("@", 1)[1]
    return f"{scheme}://<redacted>@{host}" if scheme else f"<redacted>@{host}"
