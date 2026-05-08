from __future__ import annotations

from typing import Protocol, runtime_checkable

from translation.config import TranslationConfig


@runtime_checkable
class TranslationProvider(Protocol):
    def translate_batch(self, prompt: str) -> str:
        ...

    def review_suspicious(self, prompt: str) -> str:
        ...


class OpenAICompatibleProvider(TranslationProvider):
    def __init__(self, config: TranslationConfig) -> None:
        self._config = config
        self.base_url = config.base_url
        self.model = config.model
        self.review_model = config.effective_review_model
        self.temperature = config.temperature

    def translate_batch(self, prompt: str) -> str:
        raise NotImplementedError("provider execution is not implemented in this config/CLI section")

    def review_suspicious(self, prompt: str) -> str:
        raise NotImplementedError("provider execution is not implemented in this config/CLI section")
