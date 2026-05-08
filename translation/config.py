from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


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
    output_dir: str | None = None
    dry_run: bool = False
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.provider != "openai-compatible":
            raise ValueError("only openai-compatible is supported in B1-lite")
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
            raise ValueError("qa_mode must be suspicious-only or none")

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
            "output_dir": self.output_dir,
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
        env.update(_load_env_file(Path(env_path)))

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
    if field_name in {"batch_size", "context_before", "context_after", "max_retries"}:
        return int(raw_value)
    if field_name == "temperature":
        return float(raw_value)
    if field_name == "cache_enabled":
        return _parse_bool(raw_value)
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
    host = rest.split("@", 1)[1]
    return f"{scheme}://<redacted>@{host}" if scheme else f"<redacted>@{host}"
