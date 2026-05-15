from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from translation.config import TranslationConfig
from translation.context import GlobalContext
from translation.glossary import Glossary
from translation.models import BatchState, MinimalBatchReportEntry, PipelineResult, TranslationOutputPaths
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
                fallback_provider_calls=1,
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
            self.assertIn("- report_schema_version: translation-v2-report-v2", report)
            self.assertIn("## Run Summary", report)
            self.assertIn("## Config Snapshot", report)
            self.assertIn("## Cache Summary", report)
            self.assertIn("## Provider / Fallback Summary", report)
            self.assertIn("## Concurrency Summary", report)
            self.assertIn("## Shrink-Batch Summary", report)
            self.assertIn("## QA Summary", report)
            self.assertIn("## Batch Summary", report)
            self.assertIn("## Batch Details", report)
            self.assertIn("## Terminal / Failure Summary", report)
            self.assertIn("## Warnings", report)
            self.assertIn("## Output Artifacts", report)
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
                "fallback_provider_calls": "1",
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
                    qa_reviewed=2,
                    qa_provider_calls=1,
                    qa_fixed=1,
                    qa_kept=1,
                    qa_failed=0,
                    qa_parser_failures=0,
                    qa_provider_failures=0,
                    qa_skipped=0,
                    qa_prompt_version="translation-v2-suspicious-qa-v2",
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
        self.assertIn("- qa_reviewed: 2", report)
        self.assertIn("- qa_provider_calls: 1", report)
        self.assertIn("- qa_fixed: 1", report)
        self.assertIn("- qa_kept: 1", report)
        self.assertIn("- qa_failed: 0", report)
        self.assertIn("- qa_parser_failures: 0", report)
        self.assertIn("- qa_provider_failures: 0", report)
        self.assertIn("- qa_skipped: 0", report)
        self.assertIn("- qa_prompt_version: translation-v2-suspicious-qa-v2", report)
        self.assertIn("## QA Issues", report)
        self.assertIn("- 1 | high | empty translation", report)
        self.assertIn("- 2 | medium | url count mismatch", report)
        self.assertNotIn("sk-should-not-appear", report)
        self.assertNotIn("raw", report.lower())

    def test_write_translation_report_renders_final_route_label_in_batch_results(self):
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
                cue_count=1,
                provider_called=True,
            )
            config = TranslationConfig(api_key="sk-should-not-appear")
            stats = TranslationStats(
                total_batches=1,
                provider_calls=2,
                fallback_provider_calls=1,
                batch_entries=[
                    MinimalBatchReportEntry(
                        batch_id=1,
                        state=BatchState.SUCCESS,
                        cue_count=1,
                        attempts=2,
                        cache_hit=False,
                        cue_range=(1, 1),
                        attempt=2,
                        duration_ms=12,
                        final_route_label="fallback",
                    )
                ],
            )
            glossary = Glossary(path=None, text="", hash="", exists=False, truncated=False)
            context = GlobalContext(text="", hash="")
            report_path = root / "translated" / "translation_report.md"

            write_translation_report(report_path, result, config, stats, glossary, context)

            report = report_path.read_text(encoding="utf-8")

        self.assertIn("fallback_provider_calls: 1", report)
        self.assertIn("final_route_label: fallback", report)

    def test_write_translation_report_renders_split_metadata_when_present(self):
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
            stats = TranslationStats(
                total_batches=1,
                provider_calls=3,
                batch_entries=[
                    MinimalBatchReportEntry(
                        batch_id=71,
                        state=BatchState.SUCCESS,
                        cue_count=2,
                        attempts=3,
                        cache_hit=False,
                        parent_batch_id=7,
                        child_batch_ids=(72, 73),
                        split_reason="schema_mismatch",
                        split_attempt=1,
                        split_strategy_version="v1",
                        original_target_cue_range=(1, 2),
                    )
                ],
            )
            glossary = Glossary(path=None, text="", hash="", exists=False, truncated=False)
            context = GlobalContext(text="", hash="")
            report_path = root / "translated" / "translation_report.md"

            write_translation_report(report_path, result, TranslationConfig(api_key="sk-test"), stats, glossary, context)
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("parent_batch_id: 7", report)
        self.assertIn("child_batch_ids: 72,73", report)
        self.assertIn("split_reason: schema_mismatch", report)
        self.assertIn("split_attempt: 1", report)
        self.assertIn("split_strategy_version: v1", report)
        self.assertIn("original_target_cue_range: 1-2", report)

    def test_write_translation_report_renders_stable_section_order_batch_summary_and_warnings(self):
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
                cue_count=3,
                provider_called=True,
            )
            config = TranslationConfig(api_key="sk-test", concurrency=3, qa_mode="suspicious-only")
            stats = TranslationStats(
                total_batches=3,
                provider_calls=3,
                fallback_provider_calls=1,
                cache_hits=1,
                cache_misses=2,
                retries=1,
                failed_batches=1,
                batch_entries=[
                    MinimalBatchReportEntry(
                        batch_id=7,
                        state=BatchState.SUCCESS,
                        cue_count=2,
                        attempts=3,
                        cache_hit=False,
                        cue_range=(1, 2),
                        attempt=3,
                        duration_ms=25,
                        final_route_label="fallback",
                        child_batch_ids=(71, 72),
                        split_reason="schema_mismatch",
                        split_attempt=1,
                        split_strategy_version="v1",
                        original_target_cue_range=(1, 2),
                    ),
                    MinimalBatchReportEntry(
                        batch_id=8,
                        state=BatchState.FAILED_PERMANENT,
                        cue_count=1,
                        attempts=2,
                        cache_hit=False,
                        cue_range=(3, 3),
                        attempt=2,
                        duration_ms=11,
                        final_route_label="main",
                    ),
                ],
                qa=QAStats(
                    qa_mode="suspicious-only",
                    qa_candidates=2,
                    qa_reviewed=2,
                    qa_provider_calls=1,
                    qa_fixed=1,
                    qa_kept=0,
                    qa_failed=1,
                    qa_parser_failures=0,
                    qa_provider_failures=1,
                    qa_skipped=2,
                    qa_prompt_version="translation-v2-suspicious-qa-v2",
                ),
            )
            glossary = Glossary(path=None, text="", hash="", exists=False, truncated=False)
            context = GlobalContext(text="", hash="")
            report_path = root / "translated" / "translation_report.md"

            write_translation_report(report_path, result, config, stats, glossary, context)
            report = report_path.read_text(encoding="utf-8")

        ordered_headers = [
            "## Run Summary",
            "## Config Snapshot",
            "## Cache Summary",
            "## Provider / Fallback Summary",
            "## Concurrency Summary",
            "## Shrink-Batch Summary",
            "## QA Summary",
            "## Batch Summary",
            "## Batch Details",
            "## Terminal / Failure Summary",
            "## Warnings",
            "## Output Artifacts",
        ]
        header_positions = [report.index(header) for header in ordered_headers]
        self.assertEqual(header_positions, sorted(header_positions))
        self.assertIn("- final_status: completed_with_failures", report)
        self.assertIn("- batches_with_fallback_final_route: 1", report)
        self.assertIn("- cache_hit_batches: 0", report)
        self.assertIn("- parent_batches_split: 1", report)
        self.assertIn("- child_batches_spawned: 2", report)
        self.assertIn("- split_reasons: schema_mismatch", report)
        self.assertIn("- concurrency: 3", report)
        self.assertIn("- QA failures occurred but translation-stage output was preserved.", report)
        self.assertIn("- Fallback route used for one or more batches.", report)
        self.assertIn("- Shrink-batch compensation activated for one or more parent batches.", report)


if __name__ == "__main__":
    unittest.main()
