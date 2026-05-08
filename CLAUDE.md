# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This repository is a Claude Code skill for clipping long-form videos into short segments. The main workflow is defined in `SKILL.md`: environment check, video download, subtitle analysis, chapter selection, clip processing, and output packaging.

## Common commands

Run commands from repository root: `E:/MySecondBrain/Vault/dev-workspace/projects/automation/cli/youtube-clipper-skill`.

### Environment setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Toolchain verification

```powershell
yt-dlp --version
ffmpeg -version
ffmpeg -filters | Select-String subtitles
python -c "import yt_dlp, pysrt, dotenv; print('deps ok')"
```

### Config

```powershell
Copy-Item .env.example .env
```

Important runtime settings live in `.env`, including clip pipeline settings plus translation V2 keys like `TRANSLATION_BASE_URL`, `TRANSLATION_API_KEY`, `TRANSLATION_MODEL`, `TRANSLATION_TARGET_LANG`, `TRANSLATION_QA`, and `TRANSLATION_CACHE_PATH`.

### Script entrypoints

```powershell
python scripts/download_video.py <youtube_url>
python scripts/analyze_subtitles.py <subtitle_path>
python scripts/clip_video.py <video_path> <start_time> <end_time> <output_path>
python scripts/translate_subtitles.py <subtitle_path>
python scripts/translate_subtitles_v2.py <subtitle_path> [options]
python scripts/burn_subtitles.py <video_path> <subtitle_path> <output_path>
python scripts/generate_summary.py <chapter_info>
```

### Installation as Claude skill

`install_as_skill.sh` is a bash script. On Windows, run it from Git Bash or WSL instead of plain PowerShell.

```bash
bash install_as_skill.sh
```

This installs repository into `~/.claude/skills/youtube-clipper/` for end-user skill usage.

## Architecture

### Control plane

- `SKILL.md` is source of truth for end-to-end user workflow and expected agent behavior.
- `README.md` documents installation, configuration, and output layout.
- `install_as_skill.sh` packages repo into Claude skill directory and performs dependency checks.

### Processing pipeline

Legacy skill implementation lives under `scripts/`. Translation V2 implementation is allowed under `translation/` as the B1-lite engine package.

- `download_video.py` downloads source video and subtitles with `yt-dlp`.
- `analyze_subtitles.py` parses VTT subtitles and prepares structured data for Claude chapter analysis.
- `clip_video.py` extracts video segments with FFmpeg.
- `extract_subtitle_clip.py` trims subtitle ranges to clip boundaries.
- `translate_subtitles.py` batches subtitle translation and can produce bilingual subtitle data.
- `merge_bilingual_subtitles.py` combines original and translated subtitles.
- `burn_subtitles.py` burns subtitles into video and handles FFmpeg/libass detection.
- `generate_summary.py` creates shareable summary copy for clips.
- `utils.py` holds shared helpers for time conversion, filename sanitization, path creation, and formatting.
- `translate_subtitles_v2.py` must stay thin CLI glue for config loading, pipeline invocation, and user-facing output.
- `translation/` owns reusable Translation V2 parser, provider, pipeline, cache, QA, and report modules.

### External dependencies

- `yt-dlp` handles video and subtitle acquisition.
- `ffmpeg` handles clip extraction and subtitle burn-in.
- `libass` support is required for subtitle burning.
- `pysrt` handles SRT parsing and writing.
- `python-dotenv` loads `.env` configuration.

## Repo-specific notes

- This repository is developed from the `HSJ-BanFan/Youtube-clipper-skill` fork. Development PRs should target the fork's `main` branch, not the upstream `op7418/Youtube-clipper-skill`, unless the user explicitly asks for an upstream PR.
- Repository now has focused unittest coverage for translation V2 config and CLI behavior. Other script changes still need smoke checks and dependency verification.
- Subtitle burning depends on FFmpeg subtitle filter support. If `ffmpeg -filters` does not show `subtitles`, hard-subtitle flow is broken.
- This repo is designed first as script-backed skill, not long-running service or web app. Keep changes aligned with CLI/script workflow.
- Output defaults to `./youtube-clips/<timestamp>/...` unless overridden in `.env`.
