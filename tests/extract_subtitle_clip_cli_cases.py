import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "extract_subtitle_clip.py"


class ExtractSubtitleClipCliTests(unittest.TestCase):
    def test_cli_keeps_overlapping_vtt_cues_and_writes_rebased_srt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = Path(temp_dir) / "sample.vtt"
            subtitle_path.write_text(
                "WEBVTT\n\n"
                "00:04:58.000 --> 00:05:03.000\n"
                "starts before\n\n"
                "00:05:11.000 --> 00:05:12.000\n"
                "after\n\n",
                encoding="utf-8",
            )
            output_path = Path(temp_dir) / "original.srt"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(subtitle_path),
                    "00:05:00",
                    "00:05:10",
                    str(output_path),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )

            content = output_path.read_text(encoding="utf-8") if output_path.exists() else ""

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(content, "1\n00:00:00,000 --> 00:00:03,000\nstarts before\n\n")


if __name__ == "__main__":
    unittest.main()
