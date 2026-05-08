import hashlib
import tempfile
import unittest
from pathlib import Path

from translation.config import TranslationConfig
from translation.context import build_global_context, write_global_context
from translation.models import Cue


class TranslationContextTests(unittest.TestCase):
    def test_build_global_context_includes_safe_metadata_samples_and_hash(self):
        cues = [
            Cue(id=f"cue-{index}", index=index, start="00:00:00,000", end="00:00:01,000", source=f"source line {index}")
            for index in range(1, 16)
        ]
        config = TranslationConfig(
            api_key="test-secret-key",
            target_lang="ja-JP",
            mode="publish",
            batch_size=12,
        )

        context = build_global_context(cues, Path("sample-input.srt"), config)

        self.assertIn("sample-input.srt", context.text)
        self.assertIn("cue_count", context.text)
        self.assertIn("15", context.text)
        self.assertIn("target_lang", context.text)
        self.assertIn("ja-JP", context.text)
        self.assertIn("mode", context.text)
        self.assertIn("publish", context.text)
        self.assertIn("batch_size", context.text)
        self.assertIn("12", context.text)
        self.assertIn("source line 1", context.text)
        self.assertIn("source line 5", context.text)
        self.assertIn("source line 8", context.text)
        self.assertIn("source line 11", context.text)
        self.assertIn("source line 15", context.text)
        self.assertNotIn("test-secret-key", context.text)
        self.assertEqual(len(context.hash), 64)
        self.assertEqual(context.hash, hashlib.sha256(context.text.encode("utf-8")).hexdigest())

    def test_write_global_context_writes_exact_text_to_path(self):
        cues = [Cue(id="cue-1", index=1, start="00:00:00,000", end="00:00:01,000", source="hello world")]
        context = build_global_context(cues, Path("input.srt"), TranslationConfig())

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "global-context.md"

            write_global_context(context, output_path)

            self.assertEqual(output_path.read_text(encoding="utf-8"), context.text)


if __name__ == "__main__":
    unittest.main()
