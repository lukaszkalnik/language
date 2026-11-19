# ingest

Shared library used by every tool in this repository (no command line of its own).

- **Inputs**: a local file in any format ffmpeg reads, a YouTube URL (`watch`, `youtu.be`,
  `shorts`, `live`, `embed`, with any extra query parameters), or `-` for a WAV stream on stdin.
- **YouTube**: yt-dlp (as a Python dependency, not pinned), best audio-only stream. IPv4 only with `--ipv4`, or automatically when IPv6 does not connect (`ingest.cli.choose_ipv4`).
  With `--start`/`--end` only that section is downloaded, unless fetching the whole audio and
  cutting it locally is faster: YouTube serves section downloads at ~1.7x real time but whole
  files at 50-100x, so a 15-minute section of a 2-hour video comes faster in full. Videos over
  4 hours are never fetched in full for a section; `LANGUAGE_SECTIONS_ONLY=1` always downloads
  just the section (less data). A cached whole file serves any later range without a download. Title, channel and duration go to stderr.
  Playlists, live streams and premieres are rejected; private, age-restricted, region-blocked,
  removed videos and network failures give one readable line.
- **Time ranges**: `--start` / `--end` in seconds, `mm:ss` or `hh:mm:ss`.
- **`--info`**: prints `{"type", "title", "duration", ...}` as JSON on stdout and exits before
  any download or model (YouTube: yt-dlp metadata, cached with the downloads; files: ffprobe).
  The desktop window uses it to size its time-range slider.
- **Cache**: one folder for all tools (`~/.cache/language`, or `--cache-dir` / `LANGUAGE_CACHE`),
  keyed by video ID or file hash plus time range (plus tool settings for derived audio).
- **Audio**: ffmpeg detection with an install hint, conversion that keeps the source sample
  rate unless a tool asks for a specific one, slice reading of long WAVs with flat memory.
- **CLI conventions**: data on stdout, humans on stderr, `error:` lines and exit codes,
  WAV to a file instead of a terminal.
