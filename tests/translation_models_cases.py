import unittest
from pathlib import Path

from translation.models import (
    AttemptRecord,
    BatchRecord,
    BatchState,
    Cue,
    CueRecord,
    ErrorType,
    FailureMode,
    MinimalBatchReportEntry,
    PipelineResult,
    TranslatedCue,
    TranslationBatch,
    TranslationOutputPaths,
)


class V1ModelsStillWorkTests(unittest.TestCase):
    """Verify existing V1 models remain unchanged and functional."""

    def test_cue_instantiation(self):
        cue = Cue(
            id="1",
            index=1,
            start="00:00:00,000",
            end="00:00:01,000",
            source="hello world",
        )

        self.assertEqual(cue.id, "1")
        self.assertEqual(cue.index, 1)
        self.assertEqual(cue.source, "hello world")

    def test_translated_cue_instantiation(self):
        cue = Cue(id="1", index=1, start="00:00:00,000", end="00:00:01,000", source="hello")
        translated = TranslatedCue(cue=cue, translation="你好")

        self.assertEqual(translated.cue.id, "1")
        self.assertEqual(translated.translation, "你好")

    def test_translation_batch_instantiation(self):
        cue = Cue(id="1", index=1, start="00:00:00,000", end="00:00:01,000", source="hello")
        batch = TranslationBatch(
            batch_id=1,
            cues=(cue,),
            context_before=(),
            context_after=(),
        )

        self.assertEqual(batch.batch_id, 1)
        self.assertEqual(len(batch.cues), 1)

    def test_translation_output_paths_instantiation(self):
        paths = TranslationOutputPaths(
            output_dir=Path("/tmp/output"),
            translated_srt=Path("/tmp/output/translated.srt"),
            bilingual_srt=Path("/tmp/output/bilingual.srt"),
            translation_report=Path("/tmp/output/report.json"),
            global_context=Path("/tmp/output/context.txt"),
        )

        self.assertEqual(paths.output_dir, Path("/tmp/output"))

    def test_pipeline_result_instantiation(self):
        paths = TranslationOutputPaths(
            output_dir=Path("/tmp/output"),
            translated_srt=Path("/tmp/output/translated.srt"),
            bilingual_srt=Path("/tmp/output/bilingual.srt"),
            translation_report=Path("/tmp/output/report.json"),
            global_context=Path("/tmp/output/context.txt"),
        )
        result = PipelineResult(
            input_path=Path("/tmp/input.srt"),
            input_format="srt",
            output_format="srt",
            output_paths=paths,
            dry_run=False,
            cue_count=10,
            provider_called=True,
        )

        self.assertEqual(result.cue_count, 10)
        self.assertTrue(result.provider_called)


class V2EnumsTests(unittest.TestCase):
    """Test new V2 enums for batch state and error classification."""

    def test_batch_state_enum_values(self):
        self.assertEqual(BatchState.PENDING, "pending")
        self.assertEqual(BatchState.RUNNING, "running")
        self.assertEqual(BatchState.SUCCESS, "success")
        self.assertEqual(BatchState.RETRYING, "retrying")
        self.assertEqual(BatchState.FAILED_RETRYABLE, "failed_retryable")
        self.assertEqual(BatchState.FAILED_PERMANENT, "failed_permanent")
        self.assertEqual(BatchState.SUSPICIOUS, "suspicious")
        self.assertEqual(BatchState.REPAIRING, "repairing")
        self.assertEqual(BatchState.REPAIRED, "repaired")

    def test_failure_mode_enum_values(self):
        self.assertEqual(FailureMode.STRICT, "strict")
        self.assertEqual(FailureMode.PARTIAL, "partial")
        self.assertEqual(FailureMode.INTERACTIVE, "interactive")

    def test_error_type_enum_values(self):
        self.assertEqual(ErrorType.INVALID_JSON, "invalid_json")
        self.assertEqual(ErrorType.SCHEMA_MISMATCH, "schema_mismatch")
        self.assertEqual(ErrorType.MISSING_REQUIRED_CUE_ID, "missing_required_cue_id")
        self.assertEqual(ErrorType.DUPLICATE_CUE_ID, "duplicate_cue_id")
        self.assertEqual(ErrorType.INVALID_CUE_ID, "invalid_cue_id")
        self.assertEqual(ErrorType.CONTEXT_CUE_OUTPUT_VIOLATION, "context_cue_output_violation")
        self.assertEqual(ErrorType.EMPTY_TRANSLATION, "empty_translation")


class V2CueRecordTests(unittest.TestCase):
    """Test new V2 CueRecord for passive scaffolding."""

    def test_cue_record_instantiation(self):
        record = CueRecord(
            cue_id="1",
            original_index=1,
            start="00:00:00,000",
            end="00:00:01,000",
            source_text="hello world",
        )

        self.assertEqual(record.cue_id, "1")
        self.assertEqual(record.original_index, 1)
        self.assertEqual(record.source_text, "hello world")

    def test_cue_record_with_optional_fields(self):
        record = CueRecord(
            cue_id="1",
            original_index=1,
            start="00:00:00,000",
            end="00:00:01,000",
            source_text="hello",
            raw_timing="00:00:00.000 --> 00:00:01.000",
            note="context note",
        )

        self.assertEqual(record.raw_timing, "00:00:00.000 --> 00:00:01.000")
        self.assertEqual(record.note, "context note")


class V2AttemptRecordTests(unittest.TestCase):
    """Test new V2 AttemptRecord for retry tracking."""

    def test_attempt_record_success(self):
        record = AttemptRecord(
            attempt_index=1,
            batch_id=1,
            model_alias="gpt-4",
            error_type=None,
            duration_ms=250,
            cache_hit=False,
            result_state=BatchState.SUCCESS,
        )

        self.assertEqual(record.attempt_index, 1)
        self.assertEqual(record.batch_id, 1)
        self.assertEqual(record.model_alias, "gpt-4")
        self.assertIsNone(record.error_type)
        self.assertEqual(record.duration_ms, 250)
        self.assertFalse(record.cache_hit)
        self.assertEqual(record.result_state, BatchState.SUCCESS)

    def test_attempt_record_failure(self):
        record = AttemptRecord(
            attempt_index=2,
            batch_id=1,
            model_alias="gpt-4",
            error_type=ErrorType.INVALID_JSON,
            duration_ms=100,
            cache_hit=False,
            result_state=BatchState.FAILED_RETRYABLE,
        )

        self.assertEqual(record.attempt_index, 2)
        self.assertEqual(record.error_type, ErrorType.INVALID_JSON)
        self.assertEqual(record.result_state, BatchState.FAILED_RETRYABLE)


class V2BatchRecordTests(unittest.TestCase):
    """Test new V2 BatchRecord for hierarchical batch tracking."""

    def test_batch_record_basic(self):
        record = BatchRecord(
            batch_id=1,
            target_cues=(
                CueRecord(cue_id="1", original_index=1, start="00:00:00,000", end="00:00:01,000", source_text="a"),
                CueRecord(cue_id="2", original_index=2, start="00:00:02,000", end="00:00:03,000", source_text="b"),
                CueRecord(cue_id="3", original_index=3, start="00:00:04,000", end="00:00:05,000", source_text="c"),
            ),
            context_before=(
                CueRecord(cue_id="0", original_index=0, start="00:00:00,000", end="00:00:00,500", source_text="before"),
            ),
            context_after=(
                CueRecord(cue_id="4", original_index=4, start="00:00:06,000", end="00:00:07,000", source_text="after"),
            ),
            status=BatchState.SUCCESS,
        )

        self.assertEqual(record.batch_id, 1)
        self.assertEqual(tuple(cue.cue_id for cue in record.target_cues), ("1", "2", "3"))
        self.assertEqual(tuple(cue.cue_id for cue in record.context_before), ("0",))
        self.assertEqual(tuple(cue.cue_id for cue in record.context_after), ("4",))
        self.assertEqual(record.status, BatchState.SUCCESS)

    def test_batch_record_with_split_tracking(self):
        record = BatchRecord(
            batch_id=2,
            target_cues=(
                CueRecord(cue_id="5", original_index=5, start="00:00:05,000", end="00:00:06,000", source_text="x"),
                CueRecord(cue_id="6", original_index=6, start="00:00:06,000", end="00:00:07,000", source_text="y"),
            ),
            context_before=(),
            context_after=(),
            status=BatchState.SUCCESS,
            parent_batch_id=1,
            split_reason="retry_with_smaller_batch",
            split_attempt=1,
            original_target_cue_range=(5, 10),
        )

        self.assertEqual(record.parent_batch_id, 1)
        self.assertEqual(record.split_reason, "retry_with_smaller_batch")
        self.assertEqual(record.split_attempt, 1)
        self.assertEqual(record.original_target_cue_range, (5, 10))

    def test_batch_record_with_child_batches(self):
        record = BatchRecord(
            batch_id=1,
            target_cues=(
                CueRecord(cue_id="1", original_index=1, start="00:00:01,000", end="00:00:02,000", source_text="1"),
                CueRecord(cue_id="2", original_index=2, start="00:00:02,000", end="00:00:03,000", source_text="2"),
                CueRecord(cue_id="3", original_index=3, start="00:00:03,000", end="00:00:04,000", source_text="3"),
                CueRecord(cue_id="4", original_index=4, start="00:00:04,000", end="00:00:05,000", source_text="4"),
                CueRecord(cue_id="5", original_index=5, start="00:00:05,000", end="00:00:06,000", source_text="5"),
            ),
            context_before=(),
            context_after=(),
            status=BatchState.PENDING,
            child_batch_ids=(2, 3),
        )

        self.assertEqual(len(record.child_batch_ids), 2)
        self.assertIn(2, record.child_batch_ids)
        self.assertIn(3, record.child_batch_ids)


class V2MinimalBatchReportEntryTests(unittest.TestCase):
    """Test new V2 MinimalBatchReportEntry for lightweight reporting."""

    def test_minimal_batch_report_entry(self):
        entry = MinimalBatchReportEntry(
            batch_id=1,
            state=BatchState.SUCCESS,
            cue_count=5,
            attempts=1,
            cache_hit=False,
        )

        self.assertEqual(entry.batch_id, 1)
        self.assertEqual(entry.state, BatchState.SUCCESS)
        self.assertEqual(entry.cue_count, 5)
        self.assertEqual(entry.attempts, 1)
        self.assertFalse(entry.cache_hit)

    def test_minimal_batch_report_entry_with_failure(self):
        entry = MinimalBatchReportEntry(
            batch_id=2,
            state=BatchState.FAILED_PERMANENT,
            cue_count=3,
            attempts=3,
            cache_hit=False,
            failure_mode=FailureMode.STRICT,
            error_summary="Invalid JSON response",
        )

        self.assertEqual(entry.state, BatchState.FAILED_PERMANENT)
        self.assertEqual(entry.failure_mode, FailureMode.STRICT)
        self.assertEqual(entry.error_summary, "Invalid JSON response")


class V1AndV2CoexistenceTests(unittest.TestCase):
    """Verify V1 and V2 models can be imported and used together."""

    def test_all_models_import_successfully(self):
        # V1 models
        cue = Cue(id="1", index=1, start="00:00:00,000", end="00:00:01,000", source="hello")
        translated = TranslatedCue(cue=cue, translation="你好")
        batch = TranslationBatch(batch_id=1, cues=(cue,), context_before=(), context_after=())

        # V2 models
        cue_record = CueRecord(
            cue_id="1",
            original_index=1,
            start="00:00:00,000",
            end="00:00:01,000",
            source_text="hello",
        )
        batch_record = BatchRecord(
            batch_id=1,
            target_cues=(cue_record,),
            context_before=(),
            context_after=(),
            status=BatchState.SUCCESS,
        )

        # Both should coexist without conflict
        self.assertEqual(cue.id, cue_record.cue_id)
        self.assertEqual(translated.translation, "你好")
        self.assertEqual(batch.batch_id, batch_record.batch_id)


if __name__ == "__main__":
    unittest.main()
