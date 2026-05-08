import hashlib
import tempfile
import unittest
from pathlib import Path

from translation.glossary import EMPTY_SHA256, GLOSSARY_MAX_CHARS, load_glossary


class TranslationGlossaryTests(unittest.TestCase):
    def test_missing_glossary_path_returns_empty_glossary_without_error(self):
        missing_path = Path("definitely-missing-glossary.md")

        glossary = load_glossary(missing_path)

        self.assertIsNone(glossary.path)
        self.assertEqual(glossary.text, "")
        self.assertEqual(glossary.hash, EMPTY_SHA256)
        self.assertFalse(glossary.exists)
        self.assertFalse(glossary.truncated)

    def test_none_glossary_path_returns_empty_glossary(self):
        glossary = load_glossary(None)

        self.assertIsNone(glossary.path)
        self.assertEqual(glossary.text, "")
        self.assertEqual(glossary.hash, EMPTY_SHA256)
        self.assertFalse(glossary.exists)
        self.assertFalse(glossary.truncated)

    def test_existing_glossary_loads_utf8_text_and_full_text_sha256(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            glossary_path = Path(temp_dir) / "glossary.md"
            full_text = "术语: clip\n翻译: 片段\n"
            glossary_path.write_text(full_text, encoding="utf-8")

            glossary = load_glossary(glossary_path)

        expected_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
        self.assertEqual(glossary.path, glossary_path)
        self.assertEqual(glossary.text, full_text)
        self.assertEqual(glossary.hash, expected_hash)
        self.assertTrue(glossary.exists)
        self.assertFalse(glossary.truncated)

    def test_long_glossary_truncates_injected_text_and_hashes_full_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            glossary_path = Path(temp_dir) / "long-glossary.md"
            full_text = "A" * (GLOSSARY_MAX_CHARS + 100)
            glossary_path.write_text(full_text, encoding="utf-8")

            glossary = load_glossary(glossary_path)

        expected_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
        self.assertEqual(glossary.path, glossary_path)
        self.assertEqual(glossary.text, full_text[:GLOSSARY_MAX_CHARS])
        self.assertEqual(len(glossary.text), GLOSSARY_MAX_CHARS)
        self.assertEqual(glossary.hash, expected_hash)
        self.assertTrue(glossary.exists)
        self.assertTrue(glossary.truncated)


if __name__ == "__main__":
    unittest.main()
