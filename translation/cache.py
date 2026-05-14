from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


_PHASE2_COLUMN_DEFINITIONS = (
    ("engine_version", "TEXT NOT NULL DEFAULT ''"),
    ("structured_output", "INTEGER NOT NULL DEFAULT 0"),
    ("base_url", "TEXT NOT NULL DEFAULT ''"),
    ("main_model_alias", "TEXT NOT NULL DEFAULT ''"),
    ("output_schema_version", "TEXT NOT NULL DEFAULT ''"),
    ("batching_strategy_version", "TEXT NOT NULL DEFAULT ''"),
)


@dataclass(frozen=True)
class CacheEntry:
    cache_key: str
    engine_version: str
    structured_output: bool
    provider: str
    base_url: str
    model: str
    main_model_alias: str
    target_lang: str
    prompt_version: str
    output_schema_version: str
    batching_strategy_version: str
    glossary_hash: str
    context_hash: str
    batch_source_hash: str
    result_json: str


class TranslationCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._create_table()

    def __enter__(self) -> TranslationCache:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def get(self, cache_key: str) -> str | None:
        row = self._connection.execute(
            "SELECT result_json FROM translation_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        return str(row[0])

    def set(self, entry: CacheEntry) -> None:
        self._connection.execute(
            """
            INSERT INTO translation_cache (
                cache_key,
                engine_version,
                structured_output,
                provider,
                base_url,
                model,
                main_model_alias,
                target_lang,
                prompt_version,
                output_schema_version,
                batching_strategy_version,
                glossary_hash,
                context_hash,
                batch_source_hash,
                result_json,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                engine_version = excluded.engine_version,
                structured_output = excluded.structured_output,
                provider = excluded.provider,
                base_url = excluded.base_url,
                model = excluded.model,
                main_model_alias = excluded.main_model_alias,
                target_lang = excluded.target_lang,
                prompt_version = excluded.prompt_version,
                output_schema_version = excluded.output_schema_version,
                batching_strategy_version = excluded.batching_strategy_version,
                glossary_hash = excluded.glossary_hash,
                context_hash = excluded.context_hash,
                batch_source_hash = excluded.batch_source_hash,
                result_json = excluded.result_json,
                updated_at = excluded.updated_at
            """,
            (
                entry.cache_key,
                entry.engine_version,
                int(entry.structured_output),
                entry.provider,
                entry.base_url,
                entry.model,
                entry.main_model_alias,
                entry.target_lang,
                entry.prompt_version,
                entry.output_schema_version,
                entry.batching_strategy_version,
                entry.glossary_hash,
                entry.context_hash,
                entry.batch_source_hash,
                entry.result_json,
                _utc_timestamp(),
            ),
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def _create_table(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS translation_cache (
                cache_key TEXT PRIMARY KEY,
                engine_version TEXT NOT NULL DEFAULT '',
                structured_output INTEGER NOT NULL DEFAULT 0,
                provider TEXT NOT NULL,
                base_url TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL,
                main_model_alias TEXT NOT NULL DEFAULT '',
                target_lang TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                output_schema_version TEXT NOT NULL DEFAULT '',
                batching_strategy_version TEXT NOT NULL DEFAULT '',
                glossary_hash TEXT NOT NULL,
                context_hash TEXT NOT NULL,
                batch_source_hash TEXT NOT NULL,
                result_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        existing_columns = {
            str(row[1])
            for row in self._connection.execute("PRAGMA table_info(translation_cache)").fetchall()
        }
        for column_name, column_definition in _PHASE2_COLUMN_DEFINITIONS:
            if column_name not in existing_columns:
                if not column_name.isidentifier():
                    raise ValueError(f"invalid cache column name: {column_name}")
                self._connection.execute(
                    f"ALTER TABLE translation_cache ADD COLUMN {column_name} {column_definition}"
                )
        self._connection.commit()


def build_batch_cache_key(
    engine_version: str,
    structured_output: bool,
    provider: str,
    base_url: str,
    model: str,
    main_model_alias: str,
    target_lang: str,
    prompt_version: str,
    output_schema_version: str,
    batching_strategy_version: str,
    glossary_hash: str,
    context_hash: str,
    batch_source_hash: str,
) -> str:
    payload = {
        "base_url": base_url,
        "batch_source_hash": batch_source_hash,
        "batching_strategy_version": batching_strategy_version,
        "context_hash": context_hash,
        "engine_version": engine_version,
        "glossary_hash": glossary_hash,
        "main_model_alias": main_model_alias,
        "model": model,
        "output_schema_version": output_schema_version,
        "prompt_version": prompt_version,
        "provider": provider,
        "structured_output": structured_output,
        "target_lang": target_lang,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
