from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


GLOSSARY_MAX_CHARS = 12000
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


@dataclass(frozen=True)
class Glossary:
    path: Path | None
    text: str
    hash: str
    exists: bool
    truncated: bool = False


def load_glossary(path: str | Path | None) -> Glossary:
    if path is None:
        return _empty_glossary()

    glossary_path = Path(path)
    if not glossary_path.exists():
        return _empty_glossary()

    full_text = glossary_path.read_text(encoding="utf-8")
    text = full_text[:GLOSSARY_MAX_CHARS]
    return Glossary(
        path=glossary_path,
        text=text,
        hash=hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
        exists=True,
        truncated=len(full_text) > GLOSSARY_MAX_CHARS,
    )


def _empty_glossary() -> Glossary:
    return Glossary(
        path=None,
        text="",
        hash=EMPTY_SHA256,
        exists=False,
        truncated=False,
    )
