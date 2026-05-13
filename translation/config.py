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
    "batch_max_chars": "TRANSLATION_BATCH_MAX_CHARS",
    "batch_max_cues": "TRANSLATION_BATCH_MAX_CUES",
    "output_schema_version": "TRANSLATION_OUTPUT_SCHEMA_VERSION",
    "batching_strategy_version": "TRANSLATION_BATCHING_STRATEGY_VERSION",
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
    batch_max_chars: int | None = None
    batch_max_cues: int | None = None
    output_schema_version: str = "v1"
    batching_strategy_version: str = "v1"
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
        if self.output_schema_version not in VALID_OUTPUT_SCHEMA_VERSIONS:
            valid_output_schema_versions = ", ".join(sorted(VALID_OUTPUT_SCHEMA_VERSIONS))
            raise ValueError(f"output_schema_version must be one of: {valid_output_schema_versions}")
        if self.batching_strategy_version not in VALID_BATCHING_STRATEGY_VERSIONS:
            valid_batching_strategy_versions = ", ".join(sorted(VALID_BATCHING_STRATEGY_VERSIONS))
            raise ValueError(f"batching_strategy_version must be one of: {valid_batching_strategy_versions}")
        if self.output_dir and self.output_path:
            raise ValueError("output_dir and output_path cannot both be set")

    @property
    def effective_review_model(self) -> str:
        return self.review_model or self.model

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
            "batch_max_chars": self.batch_max_chars,
            "batch_max_cues": self.batch_max_cues,
            "output_schema_version": self.output_schema_version,
            "batching_strategy_version": self.batching_strategy_version,
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
        }:
            return int(raw_value)
        if field_name == "temperature":
            return float(raw_value)
        if field_name in {"cache_enabled", "structured_output"}:
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
