import unittest

from translation.models import Cue, TranslationBatch
from translation.prompts import build_translation_prompt


def cue(cue_id, source):
    return Cue(
        id=cue_id,
        index=int(cue_id),
        start="00:00:00,000",
        end="00:00:01,000",
        source=source,
    )


class TranslationPromptTests(unittest.TestCase):
    def test_prompt_contains_target_language_and_current_cues(self):
        batch = TranslationBatch(
            batch_id=1,
            cues=[cue("1", "run python script"), cue("2", "open README.md")],
            context_before=[],
            context_after=[],
        )

        prompt = build_translation_prompt(batch, "zh-CN")

        self.assertIn("zh-CN", prompt)
        self.assertIn('"id": "1"', prompt)
        self.assertIn('"source": "run python script"', prompt)
        self.assertIn('"id": "2"', prompt)
        self.assertIn('"source": "open README.md"', prompt)

    def test_prompt_contains_before_and_after_context_as_reference_only(self):
        batch = TranslationBatch(
            batch_id=2,
            cues=[cue("3", "current cue")],
            context_before=[cue("2", "before cue")],
            context_after=[cue("4", "after cue")],
        )

        prompt = build_translation_prompt(batch, "zh-CN")

        self.assertIn("before cue", prompt)
        self.assertIn("after cue", prompt)
        self.assertIn("context", prompt.lower())
        self.assertIn("do not translate context", prompt.lower())

    def test_prompt_requires_json_only_and_strict_translation_shape(self):
        batch = TranslationBatch(
            batch_id=3,
            cues=[cue("1", "use pathlib.Path and https://example.test")],
            context_before=[],
            context_after=[],
        )

        prompt = build_translation_prompt(batch, "zh-CN")

        self.assertIn("only return JSON", prompt)
        self.assertIn("do not return Markdown", prompt)
        self.assertIn("must equal the number of cues", prompt)
        self.assertIn("id must match exactly", prompt)
        self.assertIn("translation must not be empty", prompt)
        self.assertIn("preserve code, commands, variable names, paths, URLs, and library names", prompt)
        self.assertIn('"translation"', prompt)


if __name__ == "__main__":
    unittest.main()
