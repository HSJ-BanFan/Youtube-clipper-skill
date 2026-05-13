from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from translation.config import TranslationConfig
from translation.context import GlobalContext
from translation.glossary import Glossary
from translation.models import PipelineResult, TranslationOutputPaths
from translation.qa import QAIssue
from translation.report import QAStats, TranslationStats, write_translation_report


class TranslationReportTests(unittest.TestCase):
    def test_write_translation_report_contains_safe_metadata_stats_and_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_paths = TranslationOutputPaths(
                output_dir=root / "translated",
                translated_srt=root / "translated" / "translated.zh-CN.srt",
                bilingual_srt=root / "translated" / "bilingual.srt",
                translation_report=root / "translated" / "translation_report.md",
                global_context=root / "translated" / "global_context.md",
            )
            result = PipelineResult(
                input_path=root / "input.en.srt",
                input_format="srt",
                output_format="srt",
                output_paths=output_paths,
                dry_run=False,
                cue_count=42,
                provider_called=True,
            )
            config = TranslationConfig(
                base_url="https://user:secret@example.test/v1",
                api_key="sk-should-not-appear",
                model="deepseek-chat",
                target_lang="zh-CN",
                batch_size=12,
                context_before=3,
                context_after=4,
                cache_enabled=True,
                cache_path=str(root / "cache.sqlite3"),
                glossary_path=str(root / "glossary.md"),
                engine_version="v2",
                structured_output=False,
                failure_mode="strict",
                main_model_alias="main",
                repair_model_alias="repair",
                fallback_model_alias="fallback",
                batch_max_chars=1500,
                batch_max_cues=24,
                output_schema_version="v1",
                batching_strategy_version="v1",
            )
            stats = TranslationStats(
                total_batches=5,
                provider_calls=4,
                cache_hits=1,
                cache_misses=4,
                retries=2,
                failed_batches=0,
            )
            glossary = Glossary(
                path=root / "glossary.md",
                text="CLI = 命令行界面",
                hash="glossary-hash",
                exists=True,
                truncated=True,
            )
            context = GlobalContext(
                text="global context body",
                hash="context-hash",
            )
            report_path = root / "translated" / "translation_report.md"

            write_translation_report(report_path, result, config, stats, glossary, context)

            report = report_path.read_text(encoding="utf-8")
            expected_entries = {
                "input_path": str(result.input_path),
                "input_format": "srt",
                "output_format": "srt",
                "cue_count": "42",
                "target_lang": "zh-CN",
                "model": "deepseek-chat",
                "provider": "openai-compatible",
                "batch_size": "12",
                "context_before": "3",
                "context_after": "4",
                "cache_enabled": "True",
                "cache_path": str(root / "cache.sqlite3"),
                "glossary_path": str(root / "glossary.md"),
                "engine_version": "v2",
                "structured_output": "False",
                "failure_mode": "strict",
                "main_model_alias": "main",
                "repair_model_alias": "repair",
                "fallback_model_alias": "fallback",
                "batch_max_chars": "1500",
                "batch_max_cues": "24",
                "output_schema_version": "v1",
                "batching_strategy_version": "v1",
                "glossary_hash": "glossary-hash",
                "context_hash": "context-hash",
                "total_batches": "5",
                "provider_calls": "4",
                "cache_hits": "1",
                "cache_misses": "4",
                "retries": "2",
                "failed_batches": "0",
                "output_dir": str(output_paths.output_dir),
                "translated_srt": str(output_paths.translated_srt),
                "bilingual_srt": str(output_paths.bilingual_srt),
                "translation_report": str(output_paths.translation_report),
                "global_context": str(output_paths.global_context),
                "glossary_truncated": "True",
            }
            for key, value in expected_entries.items():
                self.assertIn(f"- {key}: {value}", report)

            self.assertIn("base_url: https://<redacted>@example.test/v1", report)
            self.assertNotIn("api_key", report)
            self.assertNotIn("sk-should-not-appear", report)
            self.assertNotIn("user:secret", report)

    def test_write_translation_report_contains_qa_stats_and_issue_summary_without_raw_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_paths = TranslationOutputPaths(
                output_dir=root / "translated",
                translated_srt=root / "translated" / "translated.zh-CN.srt",
                bilingual_srt=root / "translated" / "bilingual.srt",
                translation_report=root / "translated" / "translation_report.md",
                global_context=root / "translated" / "global_context.md",
            )
            result = PipelineResult(
                input_path=root / "input.en.srt",
                input_format="srt",
                output_format="srt",
                output_paths=output_paths,
                dry_run=False,
                cue_count=2,
                provider_called=True,
            )
            config = TranslationConfig(api_key="sk-should-not-appear", qa_mode="suspicious-only")
            stats = TranslationStats(
                total_batches=1,
                provider_calls=1,
                qa=QAStats(
                    qa_mode="suspicious-only",
                    qa_candidates=2,
                    qa_provider_calls=1,
                    qa_fixed=1,
                    qa_kept=1,
                    qa_failed=0,
                    qa_prompt_version="translation-v2-suspicious-qa-v1",
                    issues=(
                        QAIssue(cue_id="1", severity="high", reason="empty translation"),
                        QAIssue(cue_id="2", severity="medium", reason="url count mismatch"),
                    ),
                ),
            )
            glossary = Glossary(path=None, text="", hash="", exists=False, truncated=False)
            context = GlobalContext(text="", hash="")
            report_path = root / "translated" / "translation_report.md"

            write_translation_report(report_path, result, config, stats, glossary, context)

            report = report_path.read_text(encoding="utf-8")

        self.assertIn("## QA", report)
        self.assertIn("- qa_mode: suspicious-only", report)
        self.assertIn("- qa_candidates: 2", report)
        self.assertIn("- qa_provider_calls: 1", report)
        self.assertIn("- qa_fixed: 1", report)
        self.assertIn("- qa_kept: 1", report)
        self.assertIn("- qa_failed: 0", report)
        self.assertIn("- qa_prompt_version: translation-v2-suspicious-qa-v1", report)
        self.assertIn("## QA Issues", report)
        self.assertIn("- 1 | high | empty translation", report)
        self.assertIn("- 2 | medium | url count mismatch", report)
        self.assertNotIn("sk-should-not-appear", report)
        self.assertNotIn("raw", report.lower())


if __name__ == "__main__":
    unittest.main()
