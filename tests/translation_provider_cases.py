import json
import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError

from translation.config import TranslationConfig
from translation.provider import OpenAICompatibleProvider, TranslationProvider


class FakeResponse:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self._payload


class TranslationProviderTests(unittest.TestCase):
    def test_openai_compatible_provider_exposes_stable_translation_interface(self):
        config = TranslationConfig(api_key="test-secret")
        provider = OpenAICompatibleProvider(config)

        self.assertIsInstance(provider, TranslationProvider)
        self.assertEqual(provider.model, "deepseek-chat")
        self.assertEqual(provider.review_model, "deepseek-chat")

    def test_provider_does_not_expose_api_key_as_public_attribute(self):
        provider = OpenAICompatibleProvider(TranslationConfig(api_key="test-secret"))

        self.assertNotIn("api_key", vars(provider))
        self.assertNotIn("test-secret", repr(provider))

    def test_translate_batch_requires_api_key(self):
        provider = OpenAICompatibleProvider(TranslationConfig(api_key=None))

        with self.assertRaisesRegex(ValueError, "TRANSLATION_API_KEY|--api-key"):
            provider.translate_batch("prompt")

    def test_translate_batch_posts_chat_completion_request(self):
        captured = {}

        def fake_urlopen(request, timeout=30):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(
                200,
                json.dumps({"choices": [{"message": {"content": "translated"}}]}).encode("utf-8"),
            )

        config = TranslationConfig(
            api_key="test-secret",
            base_url="https://api.example.test/v1/",
            model="deepseek-chat",
            temperature=0.2,
        )
        provider = OpenAICompatibleProvider(config)

        with patch("translation.provider.request.urlopen", fake_urlopen):
            result = provider.translate_batch("translate this")

        self.assertEqual(result, "translated")
        self.assertEqual(captured["url"], "https://api.example.test/v1/chat/completions")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer test-secret")
        self.assertEqual(captured["headers"]["Content-type"], "application/json")
        self.assertEqual(captured["body"]["model"], "deepseek-chat")
        self.assertEqual(captured["body"]["temperature"], 0.2)
        self.assertEqual(captured["body"]["messages"], [{"role": "user", "content": "translate this"}])

    def test_translate_batch_http_error_reports_status_without_secret(self):
        def fake_urlopen(request, timeout=30):
            raise HTTPError(
                request.full_url,
                500,
                "server error",
                hdrs=None,
                fp=BytesIO(b'{"error":"secret should stay hidden"}'),
            )

        provider = OpenAICompatibleProvider(TranslationConfig(api_key="test-secret"))

        with patch("translation.provider.request.urlopen", fake_urlopen):
            with self.assertRaisesRegex(RuntimeError, "HTTP 500") as error:
                provider.translate_batch("prompt")

        self.assertNotIn("test-secret", str(error.exception))
        self.assertNotIn("Authorization", str(error.exception))

    def test_translate_batch_rejects_missing_message_content(self):
        def fake_urlopen(request, timeout=30):
            return FakeResponse(200, json.dumps({"choices": [{}]}).encode("utf-8"))

        provider = OpenAICompatibleProvider(TranslationConfig(api_key="test-secret"))

        with patch("translation.provider.request.urlopen", fake_urlopen):
            with self.assertRaisesRegex(RuntimeError, r"choices\[0\]\.message\.content"):
                provider.translate_batch("prompt")

    def test_review_suspicious_remains_deferred_to_pr4(self):
        provider = OpenAICompatibleProvider(TranslationConfig(api_key="test-secret"))

        with self.assertRaisesRegex(NotImplementedError, "PR4"):
            provider.review_suspicious("prompt")


if __name__ == "__main__":
    unittest.main()
