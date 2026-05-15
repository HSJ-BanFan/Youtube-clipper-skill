# YouTube Clipper Skill

> AI-powered YouTube video clipper for Claude Code. Download videos, generate semantic chapters, clip segments, translate subtitles to bilingual format, and burn subtitles into videos.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

English | [简体中文](README.zh-CN.md)

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Requirements](#requirements) • [Configuration](#configuration) • [Smoke Checklist](#smoke-checklist) • [Troubleshooting](#troubleshooting) • [Release Notes](#release-notes)

---

## Features

- **AI Semantic Analysis** - Generate fine-grained chapters (2-5 minutes each) by understanding video content, not just mechanical time splitting
- **Precise Clipping** - Use FFmpeg to extract video segments with frame-accurate timing
- **Bilingual Subtitles** - Batch translate subtitles to Chinese/English with 95% API call reduction
- **Subtitle Burning** - Hardcode bilingual subtitles into videos with customizable styling
- **Content Summarization** - Auto-generate social media content (Xiaohongshu, Douyin, WeChat)

---

## Installation

### Option 1: npx skills (Recommended)

```bash
npx skills add https://github.com/op7418/Youtube-clipper-skill
```

This command will automatically install the skill to `~/.claude/skills/youtube-clipper/`.

### Option 2: Manual Installation

```bash
git clone https://github.com/op7418/Youtube-clipper-skill.git
cd Youtube-clipper-skill
bash install_as_skill.sh
```

The install script will:
- Copy files to `~/.claude/skills/youtube-clipper/`
- Install Python dependencies (yt-dlp, pysrt, python-dotenv)
- Check system dependencies (Python, yt-dlp, FFmpeg)
- Create `.env` configuration file

---

## Requirements

### System Dependencies

| Dependency | Version | Purpose | Installation |
|------------|---------|---------|--------------|
| **Python** | 3.8+ | Script execution | [python.org](https://www.python.org/downloads/) |
| **yt-dlp** | Latest | YouTube download | `brew install yt-dlp` (macOS)<br>`sudo apt install yt-dlp` (Ubuntu)<br>`pip install yt-dlp` (pip) |
| **FFmpeg with libass** | Latest | Video processing & subtitle burning | `brew install ffmpeg-full` (macOS)<br>`sudo apt install ffmpeg libass-dev` (Ubuntu) |

### Python Packages

These are automatically installed by the install script:
- `yt-dlp` - YouTube downloader
- `pysrt` - SRT subtitle parser
- `python-dotenv` - Environment variable management

### Important: FFmpeg libass Support

**macOS users**: The standard `ffmpeg` package from Homebrew does NOT include libass support (required for subtitle burning). You must install `ffmpeg-full`:

```bash
# Remove standard ffmpeg (if installed)
brew uninstall ffmpeg

# Install ffmpeg-full (includes libass)
brew install ffmpeg-full
```

**Verify libass support**:
```bash
ffmpeg -filters 2>&1 | grep subtitles
# Should output: subtitles    V->V  (...)
```

---

## Usage

### In Claude Code

Simply tell Claude to clip a YouTube video:

```
Clip this YouTube video: https://youtube.com/watch?v=VIDEO_ID
```

or

```
剪辑这个 YouTube 视频：https://youtube.com/watch?v=VIDEO_ID
```

### Workflow

1. **Environment Check** - Verifies yt-dlp, FFmpeg, and Python dependencies
2. **Video Download** - Downloads video (up to 1080p) and English subtitles
3. **AI Chapter Analysis** - Claude analyzes subtitles to generate semantic chapters (2-5 min each)
4. **User Selection** - Choose which chapters to clip and processing options
5. **Processing** - Clips video, translates subtitles, burns subtitles (if requested)
6. **Output** - Organized files in `./youtube-clips/<timestamp>/`

### Output Files

For each clipped chapter:

```
./youtube-clips/20260122_143022/
└── Chapter_Title/
    ├── Chapter_Title_clip.mp4              # Original clip (no subtitles)
    ├── Chapter_Title_with_subtitles.mp4    # With burned subtitles
    ├── Chapter_Title_bilingual.srt         # Bilingual subtitle file
    ├── translation_report.md               # Translation V2 report
    ├── global_context.md                   # Translation context summary
    └── Chapter_Title_summary.md            # Social media content
```

---

## Configuration

The skill uses environment variables for customization. Edit `~/.claude/skills/youtube-clipper/.env` or copy this repo's [.env.example](.env.example) to `.env` when running from source.

### Translation V2 core settings

| Variable | Purpose | Recommended starting point |
|----------|---------|----------------------------|
| `TRANSLATION_ENGINE_VERSION` | Selects translation pipeline generation | explicitly set `v2` (`v1` remains code default if omitted) |
| `TRANSLATION_STRUCTURED_OUTPUT` | Enables provider-side structured response mode | `false` until provider is proven compatible |
| `TRANSLATION_BASE_URL` | OpenAI-compatible endpoint base URL | provider endpoint ending in `/v1` |
| `TRANSLATION_MODEL` | Main translation model | provider default chat model |
| `TRANSLATION_FALLBACK_MODEL` | Optional fallback model for provider-pressure/error routing | leave empty unless you want fallback |
| `TRANSLATION_CONCURRENCY` | Hard upper bound for translation worker fan-out | `1` to start, raise after smoke pass |
| `TRANSLATION_ADAPTIVE_CONCURRENCY_ENABLED` | Opt-in adaptive scheduling layer | `false` for first rollout |
| `TRANSLATION_ADAPTIVE_CONCURRENCY_MIN` | Adaptive scheduler floor | `1` |
| `TRANSLATION_ADAPTIVE_CONCURRENCY_MAX` | Adaptive scheduler cap inside hard concurrency limit | empty = use `TRANSLATION_CONCURRENCY` |
| `TRANSLATION_CACHE` / `TRANSLATION_CACHE_PATH` | Enables translation-stage cache and selects cache file | `true` / `.translation_cache.sqlite3` |
| `TRANSLATION_QA` | Suspicious-only QA gate | `suspicious-only` |
| `TRANSLATION_GLOSSARY_PATH` | Optional glossary markdown path | `glossary.md` |
| `TRANSLATION_CONTEXT_BEFORE` / `TRANSLATION_CONTEXT_AFTER` | Neighbor cue context window | `10` / `10` |

### Translation V2 recommended baseline

```bash
TRANSLATION_PROVIDER=openai-compatible
TRANSLATION_BASE_URL=http://127.0.0.1:8317/v1
TRANSLATION_API_KEY=
TRANSLATION_MODEL=deepseek-chat
TRANSLATION_REVIEW_MODEL=
TRANSLATION_FALLBACK_MODEL=
TRANSLATION_TARGET_LANG=zh-CN
TRANSLATION_MODE=balanced
TRANSLATION_ENGINE_VERSION=v2
TRANSLATION_STRUCTURED_OUTPUT=false
TRANSLATION_BATCH_SIZE=80
TRANSLATION_CONTEXT_BEFORE=10
TRANSLATION_CONTEXT_AFTER=10
TRANSLATION_TEMPERATURE=0.1
TRANSLATION_MAX_RETRIES=3
TRANSLATION_FAILURE_MODE=strict
TRANSLATION_CACHE=true
TRANSLATION_CACHE_PATH=.translation_cache.sqlite3
TRANSLATION_GLOSSARY_PATH=glossary.md
TRANSLATION_QA=suspicious-only
TRANSLATION_BATCH_MAX_CHARS=
TRANSLATION_BATCH_MAX_CUES=
TRANSLATION_CONCURRENCY=1
TRANSLATION_ADAPTIVE_CONCURRENCY_ENABLED=false
TRANSLATION_ADAPTIVE_CONCURRENCY_MIN=1
TRANSLATION_ADAPTIVE_CONCURRENCY_MAX=
```

### Recommended config combinations

#### Safe default

Use for first live validation.

```bash
TRANSLATION_ENGINE_VERSION=v2
TRANSLATION_STRUCTURED_OUTPUT=false
TRANSLATION_FALLBACK_MODEL=
TRANSLATION_CONCURRENCY=1
TRANSLATION_ADAPTIVE_CONCURRENCY_ENABLED=false
TRANSLATION_QA=suspicious-only
TRANSLATION_CACHE=true
```

#### Low-cost long subtitle

Use for long inputs when throughput matters less than cost stability.

```bash
TRANSLATION_BATCH_SIZE=80
TRANSLATION_BATCH_MAX_CHARS=
TRANSLATION_BATCH_MAX_CUES=
TRANSLATION_CONCURRENCY=1
TRANSLATION_ADAPTIVE_CONCURRENCY_ENABLED=false
TRANSLATION_CACHE=true
TRANSLATION_QA=suspicious-only
```

#### Unstable provider

Use when provider shows timeouts, `429`, or `5xx` bursts.

```bash
TRANSLATION_CONCURRENCY=2
TRANSLATION_ADAPTIVE_CONCURRENCY_ENABLED=true
TRANSLATION_ADAPTIVE_CONCURRENCY_MIN=1
TRANSLATION_ADAPTIVE_CONCURRENCY_MAX=2
TRANSLATION_FALLBACK_MODEL=<fallback-model>
TRANSLATION_STRUCTURED_OUTPUT=false
```

#### High-throughput trusted provider

Use only after warm-cache and live-provider smoke checks pass.

```bash
TRANSLATION_CONCURRENCY=4
TRANSLATION_ADAPTIVE_CONCURRENCY_ENABLED=true
TRANSLATION_ADAPTIVE_CONCURRENCY_MIN=2
TRANSLATION_ADAPTIVE_CONCURRENCY_MAX=4
TRANSLATION_CACHE=true
TRANSLATION_QA=suspicious-only
```

#### Debugging / report-rich mode

Use for issue triage and report inspection.

```bash
TRANSLATION_CONCURRENCY=1
TRANSLATION_ADAPTIVE_CONCURRENCY_ENABLED=false
TRANSLATION_QA=suspicious-only
TRANSLATION_CACHE=false
TRANSLATION_FAILURE_MODE=strict
```

### Other important knobs

- `TRANSLATION_REVIEW_MODEL` defaults to `TRANSLATION_MODEL` when empty.
- `TRANSLATION_MAIN_MODEL_ALIAS`, `TRANSLATION_REPAIR_MODEL_ALIAS`, and `TRANSLATION_FALLBACK_MODEL_ALIAS` are report-facing labels; defaults are usually fine.
- `TRANSLATION_OUTPUT_SCHEMA_VERSION=v1` and `TRANSLATION_BATCHING_STRATEGY_VERSION=v1` should stay at defaults unless a future release explicitly changes them.
- `TARGET_LANGUAGE` remains old skill orchestration config. Translation V2 uses `TRANSLATION_TARGET_LANG`.

### Stable downloads / resume / optional impersonation

The downloader keeps the original yt-dlp quality target and does not default to aria2c. To make long DASH downloads more resilient, it now:
- reuses a stable per-video download directory under `OUTPUT_DIR/<extractor>_<video_id>/`
- keeps `.part` files and resumes with yt-dlp `continuedl=True`
- enables retry, fragment retry, file access retry, socket timeout, and chunked HTTP download tuning
- optionally enables yt-dlp impersonation when `YTDLP_IMPERSONATE` is set

Example optional setting:

```bash
YTDLP_IMPERSONATE=chrome:windows-10
```

Leave it empty if your environment does not need impersonation.

---

## Examples

### Example 1: Extract highlights from a tech interview

**Input**:
```
Clip this video: https://youtube.com/watch?v=Ckt1cj0xjRM
```

**Output** (AI-generated chapters):
```
1. [00:00 - 03:15] AGI as an exponential curve, not a point in time
2. [03:15 - 06:30] China's gap in AI development
3. [06:30 - 09:45] The impact of chip bans
...
```

**Result**: Select chapters → Get clipped videos with bilingual subtitles + social media content

### Example 2: Create short clips from a course

**Input**:
```
Clip this lecture video and create bilingual subtitles: https://youtube.com/watch?v=LECTURE_ID
```

**Options**:
- Generate bilingual subtitles: Yes
- Burn subtitles into video: Yes
- Generate summary: Yes

**Result**: High-quality clips ready for sharing on social media platforms

---

## Key Differentiators

### AI Semantic Chapter Analysis

Unlike mechanical time-based splitting, this skill uses Claude's AI to:
- Understand content semantics
- Identify natural topic transitions
- Generate meaningful chapter titles and summaries
- Ensure complete coverage with no gaps

**Example**:
```
❌ Mechanical splitting: [0:00-30:00], [30:00-60:00]
✅ AI semantic analysis:
   - [00:00-03:15] AGI definition
   - [03:15-07:30] China's AI landscape
   - [07:30-12:00] Chip ban impacts
```

### Batch Translation Optimization

Translates subtitle batches instead of one-by-one:
- far fewer API calls than per-cue translation
- better throughput on long subtitle files
- better translation consistency within nearby context

### Bilingual Subtitle Format

Generated subtitle files contain both English and Chinese:

```srt
1
00:00:00,000 --> 00:00:03,500
This is the English subtitle
这是中文字幕

2
00:00:03,500 --> 00:00:07,000
Another English line
另一行中文
```

---

## Smoke Checklist

Run this checklist before calling a Translation V2 setup release-ready.

### Core compatibility

- [ ] **v1 compatibility**: run the old clip flow once and confirm non-translation steps still work
- [ ] **v2 structured path**: if `TRANSLATION_STRUCTURED_OUTPUT=true`, validate one real provider run before broader rollout
- [ ] **default safe path**: validate one run with `TRANSLATION_ENGINE_VERSION=v2`, `TRANSLATION_STRUCTURED_OUTPUT=false`, `TRANSLATION_CONCURRENCY=1`

### Translation behavior checks

- [ ] **cache hit / miss**: run same subtitle twice and confirm first run shows misses and second run shows hits in `translation_report.md`
- [ ] **fallback route**: if `TRANSLATION_FALLBACK_MODEL` is configured, confirm report shows fallback usage when provider-pressure routing is triggered
- [ ] **shrink-batch**: use a long or difficult subtitle sample and confirm report records shrink metadata when split recovery activates
- [ ] **suspicious QA**: confirm `TRANSLATION_QA=suspicious-only` produces QA summary and issue list without blocking successful translation output
- [ ] **adaptive concurrency**: with adaptive mode enabled, confirm concurrency summary records low/high watermark and increase/decrease events

### Artifact review

- [ ] **report artifact review**: inspect `translation_report.md` for final status, warnings, batch details, fallback route labels, shrink metadata, and QA summary
- [ ] **global context artifact**: inspect generated `global_context.md` when present
- [ ] **bilingual output**: confirm bilingual subtitle ordering and cue alignment on a representative sample
- [ ] **long subtitle test**: run one long subtitle input end-to-end and confirm output, report, and cache artifacts remain usable

## Troubleshooting

### FFmpeg subtitle burning fails

**Error**: `Option not found: subtitles` or `filter not found`

**Solution**: Install `ffmpeg-full` (macOS) or ensure `libass-dev` is installed (Ubuntu):
```bash
# macOS
brew uninstall ffmpeg
brew install ffmpeg-full

# Ubuntu
sudo apt install ffmpeg libass-dev
```

### invalid JSON / malformed structured response

**Symptoms**:
- provider response cannot be parsed
- report shows failed batches or warnings after provider output issues

**What to check**:
- set `TRANSLATION_STRUCTURED_OUTPUT=false` unless provider is already validated for structured mode
- keep `TRANSLATION_ENGINE_VERSION=v2`
- reduce `TRANSLATION_BATCH_SIZE` or set `TRANSLATION_BATCH_MAX_CHARS` / `TRANSLATION_BATCH_MAX_CUES`
- inspect `translation_report.md` batch details and warnings before retrying broader runs

### missing `cue_id` or duplicate `cue_id`

**Symptoms**:
- batch/report output indicates schema mismatch, missing cue IDs, or duplicate cue IDs
- shrink-batch or retry activity appears in report

**What to check**:
- keep `TRANSLATION_STRUCTURED_OUTPUT=false` on unstable providers
- reduce batch size or add `TRANSLATION_BATCH_MAX_CHARS` / `TRANSLATION_BATCH_MAX_CUES`
- review `translation_report.md` for `split_reason`, `split_attempt`, and per-batch `error_type`

### provider timeout / `429` / `5xx`

**Symptoms**:
- slower translation
- retries recorded in report
- adaptive scheduler drops concurrency

**What to check**:
- lower `TRANSLATION_CONCURRENCY`
- enable adaptive mode with floor `1`
- configure `TRANSLATION_FALLBACK_MODEL` if you want fallback routing available
- keep cache enabled so successful batches are not recomputed on rerun

### fallback not used

**Why it happens**:
- `TRANSLATION_FALLBACK_MODEL` is empty
- failure did not match fallback-eligible routing conditions

**What to check**:
- set `TRANSLATION_FALLBACK_MODEL`
- inspect `translation_report.md` for `fallback_provider_calls` and `final_route_label`
- confirm the issue is provider-pressure related rather than a prompt/content problem

### cache not hit

**What to check**:
- `TRANSLATION_CACHE=true`
- same subtitle input, same config, same cache path
- cache file exists at `TRANSLATION_CACHE_PATH`
- compare `cache_hits` and `cache_misses` across two identical runs in `translation_report.md`

### adaptive concurrency not increasing / decreasing

**What to check**:
- `TRANSLATION_ADAPTIVE_CONCURRENCY_ENABLED=true`
- `TRANSLATION_CONCURRENCY` is greater than `1`
- `TRANSLATION_ADAPTIVE_CONCURRENCY_MAX` is not clamped below useful range
- review `adaptive_concurrency_*` fields in `translation_report.md`
- no pressure events means no forced decrease; short clean runs may also show no increase events

### QA failure non-fatal meaning

`qa_failed > 0` means suspicious-QA checks failed or could not repair some suspicious cues, but translation-stage output was still preserved. Review `QA Summary`, `QA Issues`, and `Warnings` in `translation_report.md` before deciding whether manual review is needed.

### report interpretation quick guide

Use `translation_report.md` as primary release-pass artifact:
- `Run Summary` → overall outcome and batch counts
- `Config Snapshot` → redacted effective config
- `Provider / Fallback Summary` → main vs fallback usage
- `Concurrency Summary` → fixed/adaptive scheduling facts only
- `Shrink-Batch Summary` and `Batch Details` → split recovery behavior
- `QA Summary` and `QA Issues` → suspicious-only QA results
- `Warnings` → why run needs extra attention

### Video download is slow

**Solution**: Set a proxy in `.env`:
```bash
YT_DLP_PROXY=http://proxy-server:port
# or
YT_DLP_PROXY=socks5://proxy-server:port
```

### Video requires login or better source access

**Symptoms**:
- age-restricted or region-restricted video
- lower-quality stream than expected
- metadata or download access blocked

**Solution**: use browser cookies or a cookies file in `.env`:
```bash
YT_DLP_COOKIES_FROM_BROWSER=firefox
# or
YT_DLP_COOKIES_FILE=/path/to/cookies.txt
```

Use only one cookie source at a time.

### Special characters in filenames

**Issue**: Filenames with `:`, `/`, `?`, etc. may cause errors

**Solution**: The skill automatically sanitizes filenames by:
- Removing special characters: `/ \ : * ? " < > |`
- Replacing spaces with underscores
- Limiting length to 100 characters

---

## Release Notes

### Phase 0-8 summary

Translation V2 now includes:
- Phase 0/1 foundation and CLI integration
- cache-backed translation reuse
- bounded concurrency
- provider error routing
- shrink-batch recovery
- suspicious-only QA
- expanded translation reporting
- opt-in adaptive concurrency scheduling

### Known limitations

- provider compatibility for structured output still needs real-provider validation
- adaptive concurrency is scheduling-only; it does not improve bad provider output quality
- smoke validation should be repeated when changing provider, model, base URL, or fallback model
- report review is still required for release-critical subtitle runs

### Explicitly unsupported in current release pass

- partial-success workflow redesign
- new provider-routing framework
- new repair mode
- QA rewrite
- cache-key redesign
- prompt rewrite
- adaptive algorithm redesign
- full pipeline rewrite

---

## Documentation

- **[SKILL.md](SKILL.md)** - Complete workflow and operator-facing skill behavior
- **[.env.example](.env.example)** - Full configuration template for local `.env`
- **[TECHNICAL_NOTES.md](TECHNICAL_NOTES.md)** - Implementation notes and design decisions
- **[FIXES_AND_IMPROVEMENTS.md](FIXES_AND_IMPROVEMENTS.md)** - Historical fixes and improvements
- **[references/](references/)** - FFmpeg, yt-dlp, and subtitle formatting guides

---

## Contributing

Contributions are welcome! Please:
- Report bugs via [GitHub Issues](https://github.com/op7418/Youtube-clipper-skill/issues)
- Submit feature requests
- Open pull requests for improvements

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **[Claude Code](https://claude.ai/claude-code)** - The AI-powered CLI tool
- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** - YouTube download engine
- **[FFmpeg](https://ffmpeg.org/)** - Video processing powerhouse

---

<div align="center">

**Made with ❤️ by [op7418](https://github.com/op7418)**

If this skill helps you, please give it a ⭐️

</div>
