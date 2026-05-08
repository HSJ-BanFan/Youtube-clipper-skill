import unittest

from translation.config import TranslationConfig
from translation.provider import OpenAICompatibleProvider, TranslationProvider


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

    def test_provider_placeholder_fails_clearly_without_calling_network(self):
        provider = OpenAICompatibleProvider(TranslationConfig(api_key="test-secret"))

        with self.assertRaisesRegex(NotImplementedError, "not implemented in this config/CLI section"):
            provider.translate_batch("prompt")

        with self.assertRaisesRegex(NotImplementedError, "not implemented in this config/CLI section"):
            provider.review_suspicious("prompt")


if __name__ == "__main__":
    unittest.main()
