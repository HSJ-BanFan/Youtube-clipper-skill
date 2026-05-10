import unittest

from translation.models import Cue
from translation.qa import find_suspicious_translations


def cue(cue_id, source):
    return Cue(
        id=cue_id,
        index=int(cue_id),
        start="00:00:00,000",
        end="00:00:01,000",
        source=source,
    )


def reasons_for(candidates, cue_id):
    for candidate in candidates:
        if candidate.cue.id == cue_id:
            return tuple(issue.reason for issue in candidate.issues)
    return ()


class SuspiciousTranslationRuleTests(unittest.TestCase):
    def test_empty_translation_is_marked(self):
        cues = [cue("1", "hello world")]

        candidates = find_suspicious_translations(cues, {"1": "   "}, "zh-CN")

        self.assertEqual([candidate.cue.id for candidate in candidates], ["1"])
        self.assertIn("empty translation", reasons_for(candidates, "1"))

    def test_json_or_markdown_leak_is_marked(self):
        cues = [cue("1", "translate this sentence")]

        candidates = find_suspicious_translations(
            cues,
            {"1": '```json\n[{"id":"1","translation":"你好"}]\n```'},
            "zh-CN",
        )

        self.assertIn("json or markdown leak", reasons_for(candidates, "1"))

    def test_ai_refusal_is_marked(self):
        cues = [cue("1", "explain how this CLI works")]

        candidates = find_suspicious_translations(cues, {"1": "As an AI, I cannot help."}, "zh-CN")

        self.assertIn("model refusal", reasons_for(candidates, "1"))

    def test_url_count_mismatch_is_marked(self):
        cues = [cue("1", "open https://example.test/docs and read it")]

        candidates = find_suspicious_translations(cues, {"1": "打开文档并阅读"}, "zh-CN")

        self.assertIn("url count mismatch", reasons_for(candidates, "1"))

    def test_zh_cn_long_english_source_without_chinese_is_marked(self):
        cues = [cue("1", "This command configures the translation pipeline and writes subtitle output files.")]

        candidates = find_suspicious_translations(
            cues,
            {"1": "This command configures translation pipeline."},
            "zh-CN",
        )

        self.assertIn("missing target-language characters", reasons_for(candidates, "1"))

    def test_ko_long_english_source_with_korean_is_not_marked_missing_target_script(self):
        cues = [cue("1", "This command configures the translation pipeline and writes subtitle output files.")]

        candidates = find_suspicious_translations(
            cues,
            {"1": "이 명령은 번역 파이프라인을 설정하고 자막 출력 파일을 작성합니다."},
            "ko",
        )

        self.assertNotIn("missing target-language characters", reasons_for(candidates, "1"))

    def test_pure_code_and_url_cues_are_not_forced_to_target_language(self):
        cues = [
            cue("1", "python scripts/translate_subtitles_v2.py --help"),
            cue("2", "https://example.test/docs"),
        ]

        candidates = find_suspicious_translations(
            cues,
            {
                "1": "python scripts/translate_subtitles_v2.py --help",
                "2": "https://example.test/docs",
            },
            "zh-CN",
        )

        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()
