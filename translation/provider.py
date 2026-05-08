from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

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
        if not self._config.api_key:
            raise ValueError("TRANSLATION_API_KEY is required. Set it as an environment variable or provide it via --env-file.")

        response_data = self._post_chat_completion(prompt)
        return _extract_message_content(response_data)

    def review_suspicious(self, prompt: str) -> str:
        raise NotImplementedError("suspicious review is deferred to PR4")

    def _post_chat_completion(self, prompt: str) -> Any:
        parsed_base_url = urlparse(self.base_url)
        if parsed_base_url.scheme not in {"http", "https"} or not parsed_base_url.netloc:
            raise ValueError("base_url must use http:// or https://")

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=30) as response:
                status = response.status
                if status < 200 or status >= 300:
                    raise RuntimeError(f"translation provider HTTP {status}")
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            raise RuntimeError(f"translation provider HTTP {exc.code}") from None
        except URLError:
            raise RuntimeError("translation provider request failed") from None

        try:
            return json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("translation provider returned invalid JSON") from exc


def _extract_message_content(response_data: Any) -> str:
    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("translation provider response missing choices[0].message.content") from exc
    if not isinstance(content, str):
        raise RuntimeError("translation provider response missing choices[0].message.content")
    return content
