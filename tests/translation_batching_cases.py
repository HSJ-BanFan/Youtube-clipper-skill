import unittest

from translation.batching import create_batches
from translation.models import Cue


def make_cues(count):
    return [
        Cue(
            id=str(index),
            index=index,
            start=f"00:00:0{index},000",
            end=f"00:00:0{index},500",
            source=f"source {index}",
        )
        for index in range(1, count + 1)
    ]


class TranslationBatchingTests(unittest.TestCase):
    def test_create_batches_splits_translation_cues_without_context(self):
        cues = make_cues(5)

        batches = create_batches(cues, batch_size=2, context_before=1, context_after=1)

        self.assertEqual(len(batches), 3)
        self.assertEqual([batch.batch_id for batch in batches], [1, 2, 3])
        self.assertEqual([[cue.id for cue in batch.cues] for batch in batches], [["1", "2"], ["3", "4"], ["5"]])

    def test_create_batches_sets_context_windows(self):
        cues = make_cues(5)

        batches = create_batches(cues, batch_size=2, context_before=2, context_after=2)

        self.assertEqual(batches[0].context_before, ())
        self.assertEqual([cue.id for cue in batches[0].context_after], ["3", "4"])
        self.assertEqual([cue.id for cue in batches[1].context_before], ["1", "2"])
        self.assertEqual([cue.id for cue in batches[1].context_after], ["5"])
        self.assertEqual([cue.id for cue in batches[2].context_before], ["3", "4"])
        self.assertEqual(batches[2].context_after, ())

    def test_context_cues_do_not_enter_translation_cues(self):
        cues = make_cues(5)

        batches = create_batches(cues, batch_size=2, context_before=2, context_after=2)

        self.assertEqual([cue.id for cue in batches[1].cues], ["3", "4"])

    def test_create_batches_rejects_zero_batch_size(self):
        with self.assertRaisesRegex(ValueError, "batch_size must be greater than 0"):
            create_batches(make_cues(1), batch_size=0, context_before=0, context_after=0)

    def test_create_batches_rejects_negative_context_before(self):
        with self.assertRaisesRegex(
            ValueError,
            "context_before and context_after must be greater than or equal to 0",
        ):
            create_batches(make_cues(1), batch_size=1, context_before=-1, context_after=0)

    def test_create_batches_rejects_negative_context_after(self):
        with self.assertRaisesRegex(
            ValueError,
            "context_before and context_after must be greater than or equal to 0",
        ):
            create_batches(make_cues(1), batch_size=1, context_before=0, context_after=-1)


if __name__ == "__main__":
    unittest.main()
