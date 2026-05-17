from __future__ import annotations

import json
import socket
from typing import Any, Protocol, runtime_checkable
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

from translation.config import TranslationConfig
from translation.models import ErrorType

REQUEST_TIMEOUT_SECONDS = 120


class ProviderError(RuntimeError):
    def __init__(self, error_type: ErrorType, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.cause = cause


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

        response_data = self._post_chat_completion(prompt, self.model)
        return _extract_message_content(response_data)

    def review_suspicious(self, prompt: str) -> str:
        if not self._config.api_key:
            raise ValueError("TRANSLATION_API_KEY is required. Set it as an environment variable or provide it via --env-file.")

        response_data = self._post_chat_completion(prompt, self.review_model)
        return _extract_message_content(response_data)

    def segment_semantically(self, prompt: str, model: str | None = None) -> str:
        if not self._config.api_key:
            raise ValueError("TRANSLATION_API_KEY is required. Set it as an environment variable or provide it via --env-file.")

        response_data = self._post_chat_completion(prompt, model or self.model)
        return _extract_message_content(response_data)

    def _post_chat_completion(self, prompt: str, model: str) -> Any:
        parsed_base_url = urlparse(self.base_url)
        if parsed_base_url.scheme not in {"http", "https"} or not parsed_base_url.netloc:
            raise ValueError("base_url must use http:// or https://")

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "stream": False,
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
            with request.urlopen(http_request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                status = response.status
                if status < 200 or status >= 300:
                    error_type = ErrorType.PROVIDER_HTTP_5XX if 500 <= status < 600 else ErrorType.PROVIDER_REQUEST_FAILED
                    raise ProviderError(error_type, f"translation provider HTTP {status}")
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            error_type = ErrorType.PROVIDER_HTTP_5XX if 500 <= exc.code < 600 else ErrorType.PROVIDER_REQUEST_FAILED
            raise ProviderError(error_type, f"translation provider HTTP {exc.code}", cause=exc) from None
        except (TimeoutError, socket.timeout) as exc:
            raise ProviderError(ErrorType.PROVIDER_TIMEOUT, "translation provider request timed out", cause=exc) from None
        except URLError as exc:
            raise ProviderError(ErrorType.PROVIDER_REQUEST_FAILED, "translation provider request failed", cause=exc) from None

        return _load_response_json(response_body)


def _load_response_json(response_body: str) -> Any:
    decoder = json.JSONDecoder()
    stripped_body = response_body.lstrip()
    try:
        response_data, end_index = decoder.raw_decode(stripped_body)
    except json.JSONDecodeError as exc:
        raise ProviderError(ErrorType.PROVIDER_REQUEST_FAILED, "translation provider returned invalid JSON", cause=exc) from exc

    trailing = stripped_body[end_index:].strip()
    if trailing and not _is_allowed_sse_trailer(trailing):
        raise ProviderError(ErrorType.PROVIDER_REQUEST_FAILED, "translation provider returned invalid JSON")
    return response_data


def _is_allowed_sse_trailer(trailing: str) -> bool:
    return all(line == "data: [DONE]" for line in trailing.splitlines() if line.strip())


def _extract_message_content(response_data: Any) -> str:
    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(
            ErrorType.PROVIDER_MISSING_CHOICES,
            "translation provider response missing choices[0].message.content",
            cause=exc,
        ) from exc
    if not isinstance(content, str):
        raise ProviderError(
            ErrorType.PROVIDER_MISSING_CHOICES,
            "translation provider response missing choices[0].message.content",
        )
    return content
