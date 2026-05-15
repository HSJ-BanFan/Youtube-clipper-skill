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

    def test_numeric_mismatch_is_marked(self):
        cues = [cue("1", "Retry 3 times after 15 seconds.")]

        candidates = find_suspicious_translations(cues, {"1": "在 5 秒后重试一次。"}, "zh-CN")

        self.assertIn("numeric token mismatch", reasons_for(candidates, "1"))

    def test_unit_mismatch_is_marked(self):
        cues = [cue("1", "Set timeout to 500 ms before retry.")]

        candidates = find_suspicious_translations(cues, {"1": "将超时设置为 500 秒后重试。"}, "zh-CN")

        self.assertIn("unit token mismatch", reasons_for(candidates, "1"))

    def test_windows_path_mismatch_is_marked(self):
        cues = [cue("1", r"Open C:\clips\input.srt before running tool.")]

        candidates = find_suspicious_translations(cues, {"1": "运行工具前打开输入文件。"}, "zh-CN")

        self.assertIn("path token mismatch", reasons_for(candidates, "1"))

    def test_posix_path_mismatch_is_marked(self):
        cues = [cue("1", "Copy /var/tmp/output.srt into /srv/archive/output.srt now.")]

        candidates = find_suspicious_translations(cues, {"1": "现在复制输出文件。"}, "zh-CN")

        self.assertIn("path token mismatch", reasons_for(candidates, "1"))

    def test_cli_flag_mismatch_is_marked(self):
        cues = [cue("1", "Run script with --dry-run and --output result.srt.")]

        candidates = find_suspicious_translations(cues, {"1": "运行脚本并输出结果。"}, "zh-CN")

        self.assertIn("option token mismatch", reasons_for(candidates, "1"))

    def test_env_var_mismatch_is_marked(self):
        cues = [cue("1", "Set TRANSLATION_API_KEY before running command.")]

        candidates = find_suspicious_translations(cues, {"1": "运行命令前先设置密钥。"}, "zh-CN")

        self.assertIn("env var mismatch", reasons_for(candidates, "1"))

    def test_polluted_output_prefix_is_marked(self):
        cues = [cue("1", "Ship release now.")]

        candidates = find_suspicious_translations(cues, {"1": "Translation: 立即发布版本。"}, "zh-CN")

        self.assertIn("polluted output", reasons_for(candidates, "1"))

    def test_short_technical_cue_can_still_be_marked_unusually_short(self):
        cues = [cue("1", "Stream copy keeps codec unchanged in FFmpeg.")]

        candidates = find_suspicious_translations(cues, {"1": "保持。"}, "zh-CN")

        self.assertIn("translation unusually short", reasons_for(candidates, "1"))

    def test_reordered_preserved_tokens_do_not_trigger_suspicious(self):
        cues = [
            cue(
                "1",
                "Open https://a.test https://b.test run python tool.py --dry-run --output /srv/out.srt after 5 s set TRANSLATION_API_KEY check C:\\clips\\input.srt with ffmpeg()",
            )
        ]

        candidates = find_suspicious_translations(
            cues,
            {
                "1": "先设置 TRANSLATION_API_KEY 再打开 https://b.test 和 https://a.test 然后用 ffmpeg() 检查 C:\\clips\\input.srt 并在 5 s 后运行 python tool.py --output /srv/out.srt --dry-run",
            },
            "zh-CN",
        )

        self.assertEqual(candidates, [])

    def test_missing_preserved_tokens_still_trigger_suspicious(self):
        cues = [
            cue(
                "1",
                "Open https://a.test then https://b.test, run python tool.py --dry-run --output /srv/out.srt after 5 s, set TRANSLATION_API_KEY, and check C:\\clips\\input.srt with ffmpeg().",
            )
        ]

        candidates = find_suspicious_translations(
            cues,
            {
                "1": "打开 https://a.test，然后在 5 s 后运行 python tool.py --output /srv/out.srt。",
            },
            "zh-CN",
        )

        self.assertIn("url count mismatch", reasons_for(candidates, "1"))
        self.assertIn("path token mismatch", reasons_for(candidates, "1"))
        self.assertIn("option token mismatch", reasons_for(candidates, "1"))
        self.assertIn("env var mismatch", reasons_for(candidates, "1"))
        self.assertIn("code token loss", reasons_for(candidates, "1"))

    def test_duplicated_or_mismatched_token_counts_still_trigger_suspicious(self):
        cues = [
            cue(
                "1",
                "Open https://a.test, run python tool.py --dry-run after 5 s, set TRANSLATION_API_KEY, and inspect C:\\clips\\input.srt.",
            )
        ]

        candidates = find_suspicious_translations(
            cues,
            {
                "1": "打开 https://a.test 和 https://a.test，在 6 s 后运行 python tool.py --dry-run --dry-run，并设置 TRANSLATION_API_KEY TRANSLATION_API_KEY，再检查 C:\\clips\\other.srt。",
            },
            "zh-CN",
        )

        self.assertIn("url count mismatch", reasons_for(candidates, "1"))
        self.assertIn("numeric token mismatch", reasons_for(candidates, "1"))
        self.assertIn("path token mismatch", reasons_for(candidates, "1"))
        self.assertIn("option token mismatch", reasons_for(candidates, "1"))
        self.assertIn("env var mismatch", reasons_for(candidates, "1"))
        self.assertIn("code token loss", reasons_for(candidates, "1"))


if __name__ == "__main__":
    unittest.main()
