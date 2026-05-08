from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class CacheEntry:
    cache_key: str
    provider: str
    model: str
    target_lang: str
    prompt_version: str
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
                provider,
                model,
                target_lang,
                prompt_version,
                glossary_hash,
                context_hash,
                batch_source_hash,
                result_json,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                provider = excluded.provider,
                model = excluded.model,
                target_lang = excluded.target_lang,
                prompt_version = excluded.prompt_version,
                glossary_hash = excluded.glossary_hash,
                context_hash = excluded.context_hash,
                batch_source_hash = excluded.batch_source_hash,
                result_json = excluded.result_json,
                updated_at = excluded.updated_at
            """,
            (
                entry.cache_key,
                entry.provider,
                entry.model,
                entry.target_lang,
                entry.prompt_version,
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
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                glossary_hash TEXT NOT NULL,
                context_hash TEXT NOT NULL,
                batch_source_hash TEXT NOT NULL,
                result_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()


def build_batch_cache_key(
    provider: str,
    model: str,
    target_lang: str,
    prompt_version: str,
    glossary_hash: str,
    context_hash: str,
    batch_source_hash: str,
) -> str:
    payload = {
        "batch_source_hash": batch_source_hash,
        "context_hash": context_hash,
        "glossary_hash": glossary_hash,
        "model": model,
        "prompt_version": prompt_version,
        "provider": provider,
        "target_lang": target_lang,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
