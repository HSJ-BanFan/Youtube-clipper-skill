import unittest

from translation.models import Cue, TranslationBatch
from translation.prompts import (
    PROMPT_VERSION,
    QA_PROMPT_VERSION,
    build_suspicious_qa_prompt,
    build_translation_prompt,
)
from translation.qa import QACandidate, QAIssue


def cue(cue_id, source):
    return Cue(
        id=cue_id,
        index=int(cue_id),
        start="00:00:00,000",
        end="00:00:01,000",
        source=source,
    )


class TranslationPromptTests(unittest.TestCase):
    def test_prompt_version_is_stable_for_cache_and_reports(self):
        self.assertEqual(PROMPT_VERSION, "translation-v2-json-cue-v1")

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

    def test_prompt_includes_glossary_and_global_context_sections(self):
        batch = TranslationBatch(
            batch_id=4,
            cues=[cue("1", "ship the release")],
            context_before=[],
            context_after=[],
        )

        prompt = build_translation_prompt(
            batch,
            "zh-CN",
            glossary_text="release = 发布版本",
            global_context_text="This video explains a Python CLI release workflow.",
        )

        self.assertIn("Glossary", prompt)
        self.assertIn("release = 发布版本", prompt)
        self.assertIn("Global Context", prompt)
        self.assertIn("This video explains a Python CLI release workflow.", prompt)
        self.assertIn("glossary", prompt.lower())
        self.assertIn("follow", prompt.lower())
        self.assertIn("consistently", prompt.lower())

    def test_prompt_uses_empty_arrays_for_missing_glossary_and_global_context(self):
        batch = TranslationBatch(
            batch_id=5,
            cues=[cue("1", "ship the release")],
            context_before=[],
            context_after=[],
        )

        prompt = build_translation_prompt(batch, "zh-CN")

        self.assertIn("Glossary:\n[]", prompt)
        self.assertIn("Global Context:\n[]", prompt)

    def test_prompt_says_not_to_translate_context_or_global_context(self):
        batch = TranslationBatch(
            batch_id=6,
            cues=[cue("2", "current cue")],
            context_before=[cue("1", "before cue")],
            context_after=[cue("3", "after cue")],
        )

        prompt = build_translation_prompt(
            batch,
            "zh-CN",
            glossary_text="current cue = 当前字幕",
            global_context_text="Use this for topic understanding only.",
        ).lower()

        self.assertIn("do not translate context", prompt)
        self.assertIn("do not translate global context", prompt)


class SuspiciousQAPromptTests(unittest.TestCase):
    def test_qa_prompt_version_is_stable_for_reports(self):
        self.assertEqual(QA_PROMPT_VERSION, "translation-v2-suspicious-qa-v1")

    def test_qa_prompt_contains_candidate_id_source_translation_and_issues(self):
        candidate = QACandidate(
            cue=cue("7", "open https://example.test/docs"),
            translation="打开文档",
            issues=(QAIssue(cue_id="7", severity="medium", reason="url count mismatch"),),
        )

        prompt = build_suspicious_qa_prompt([candidate], "zh-CN")

        self.assertIn('"id": "7"', prompt)
        self.assertIn('"source": "open https://example.test/docs"', prompt)
        self.assertIn('"translation": "打开文档"', prompt)
        self.assertIn('"severity": "medium"', prompt)
        self.assertIn('"reason": "url count mismatch"', prompt)

    def test_qa_prompt_requires_json_only_and_keep_or_fix_actions(self):
        candidate = QACandidate(
            cue=cue("1", "hello world"),
            translation="你好世界",
            issues=(QAIssue(cue_id="1", severity="low", reason="translation unusually short"),),
        )

        prompt = build_suspicious_qa_prompt([candidate], "zh-CN")

        self.assertIn("only return JSON", prompt)
        self.assertIn("do not return Markdown", prompt)
        self.assertIn('"action": "keep" | "fix"', prompt)
        self.assertIn("Only fix obvious errors", prompt)
        self.assertIn("do not add, delete, or reorder ids", prompt)

    def test_qa_prompt_includes_glossary_and_global_context_without_outputting_context(self):
        candidate = QACandidate(
            cue=cue("2", "ship the release"),
            translation="发布版本",
            issues=(QAIssue(cue_id="2", severity="medium", reason="translation unusually short"),),
        )

        prompt = build_suspicious_qa_prompt(
            [candidate],
            "zh-CN",
            glossary_text="release = 发布版本",
            global_context_text="This video explains a Python CLI release workflow.",
        )

        self.assertIn("Glossary", prompt)
        self.assertIn("release = 发布版本", prompt)
        self.assertIn("Global Context", prompt)
        self.assertIn("This video explains a Python CLI release workflow.", prompt)
        self.assertIn("do not translate global context", prompt.lower())
        self.assertIn("do not output global context", prompt.lower())


if __name__ == "__main__":
    unittest.main()
