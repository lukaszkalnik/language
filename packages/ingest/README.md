# ingest

Shared library used by every tool in this repository (no command line of its own).

- **Inputs**: a local file in any format ffmpeg reads, a YouTube URL (`watch`, `youtu.be`,
  `shorts`, `live`, `embed`, with any extra query parameters), or `-` for a WAV stream on stdin.
- **YouTube**: yt-dlp (as a Python dependency, not pinned), best audio-only stream; with
  `--start`/`--end` only that section is downloaded. Title, channel and duration go to stderr.
  Playlists, live streams and premieres are rejected; private, age-restricted, region-blocked,
  removed videos and network failures give one readable line.
- **Time ranges**: `--start` / `--end` in seconds, `mm:ss` or `hh:mm:ss`.
- **Cache**: one folder for all tools (`~/.cache/language`, or `--cache-dir` / `LANGUAGE_CACHE`),
  keyed by video ID or file hash plus time range (plus tool settings for derived audio).
- **Audio**: ffmpeg detection with an install hint, conversion that keeps the source sample
  rate unless a tool asks for a specific one, slice reading of long WAVs with flat memory.
- **CLI conventions**: data on stdout, humans on stderr, `error:` lines and exit codes,
  WAV to a file instead of a terminal.
