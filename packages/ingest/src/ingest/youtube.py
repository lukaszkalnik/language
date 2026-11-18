"""YouTube ingest with yt-dlp: URL parsing, metadata, audio-only section downloads, caching."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .cache import cache_key
from .errors import YouTubeError

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_YOUTUBE_HOSTS = ("youtube.com", "youtube-nocookie.com", "music.youtube.com")
UPGRADE_HINT = "YouTube may have changed; try upgrading yt-dlp (uv lock --upgrade-package yt-dlp && uv sync)"


# --- URL handling (no network) ---------------------------------------------------

def looks_like_url(arg: str) -> bool:
    text = arg.strip().lower()
    if text.startswith(("http://", "https://")):
        return True
    host = text.split("/", 1)[0]
    return host.endswith(("youtube.com", "youtu.be", "youtube-nocookie.com")) and "/" in text


def _host_matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def parse_video_id(url: str) -> str:
    """Return the 11-character video ID of a single-video YouTube URL.

    Accepts watch, youtu.be, shorts, live and embed URLs with any extra query
    parameters. Raises YouTubeError for playlists, channels and other sites.
    """
    text = url.strip()
    if not text.lower().startswith(("http://", "https://")):
        text = "https://" + text
    parts = urlsplit(text)
    host = (parts.hostname or "").lower()
    query = parse_qs(parts.query)
    segments = [s for s in parts.path.split("/") if s]

    video_id = None
    if _host_matches(host, "youtu.be"):
        video_id = segments[0] if segments else None
    elif any(_host_matches(host, d) for d in _YOUTUBE_HOSTS):
        first = segments[0].lower() if segments else ""
        if first == "watch":
            video_id = (query.get("v") or [None])[0]
        elif first in ("shorts", "live", "embed", "v", "e") and len(segments) > 1:
            video_id = segments[1]
        if video_id is None and ("list" in query or first == "playlist"):
            raise YouTubeError("Playlists are not supported; pass the URL of a single video.")
    else:
        raise YouTubeError(f"Only YouTube video URLs are supported (got {parts.hostname or url!r}).")

    if not video_id or not _ID_RE.match(video_id):
        raise YouTubeError(f"Could not find a video ID in {url!r}; pass a single-video YouTube URL.")
    return video_id


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


# --- error mapping (no network) --------------------------------------------------

_NETWORK_MARKERS = (
    "unable to download webpage", "unable to download api page", "getaddrinfo failed",
    "nodename nor servname", "name or service not known", "temporary failure in name resolution",
    "network is unreachable", "no route to host", "connection refused", "connection reset",
    "timed out", "urlopen error", "transporterror", "failed to resolve",
)


def explain_error(message: str) -> str:
    """Turn a yt-dlp error message into one readable line."""
    clean = _ANSI_RE.sub("", message).strip()
    clean = re.sub(r"^ERROR:\s*", "", clean)
    low = clean.lower()
    if "private video" in low or "video is private" in low:
        return "This video is private."
    if "confirm your age" in low or "age-restricted" in low or "inappropriate for some users" in low:
        return "This video is age-restricted and cannot be downloaded without signing in."
    if any(m in low for m in ("available in your country", "geo restrict", "geo-restrict", "blocked it in your country")):
        return "This video is blocked in your region."
    if "live event will begin" in low or "premieres in" in low or "is live" in low:
        return "Live streams and upcoming premieres are not supported; pass a finished video."
    if any(m in low for m in ("removed", "terminated", "video unavailable", "video is unavailable",
                              "does not exist", "no longer available")):
        return "This video is unavailable (removed, deleted or never existed)."
    if "not a bot" in low:
        return f"YouTube asked to confirm this is not a bot and refused the download. {UPGRADE_HINT}"
    if any(marker in low for marker in _NETWORK_MARKERS):
        return "Network error: could not reach YouTube. Check your connection (or try --ipv4)."
    if "requested format is not available" in low:
        return "No audio-only stream is available for this video."
    first_line = clean.splitlines()[0] if clean else "unknown error"
    return f"YouTube download failed: {first_line}. {UPGRADE_HINT}"


# --- metadata, range and caching ----------------------------------------------------

@dataclass
class VideoInfo:
    id: str
    title: str
    channel: str
    duration: float | None  # seconds; None if YouTube does not report it


@dataclass
class Download:
    path: Path
    info: VideoInfo
    start: float  # where the downloaded audio begins in the original video
    end: float | None  # where it ends (None: end of video)
    capped: bool  # True when a default cap (e.g. accent's 10 minutes) was applied
    from_cache: bool


def plan_range(
    start: float | None,
    end: float | None,
    duration: float | None,
    cap: float | None = None,
) -> tuple[float, float | None, bool]:
    """Decide which part of the video to fetch. Returns (start, end, capped).

    `cap` limits a request without --start/--end to the first `cap` seconds.
    """
    capped = False
    if start is None and end is None and cap is not None:
        if duration is None or duration > cap:
            end, capped = cap, True
    begin = start or 0.0
    if duration is not None:
        if begin >= duration:
            raise YouTubeError(f"--start is past the end of the video ({duration:.0f} s long).")
        if end is not None and end >= duration:
            end = None  # same as "to the end"; keeps the cache key stable
            capped = False
    return begin, end, capped


_SKIP_SUFFIXES = (".part", ".ytdl", ".json", ".tmp")


def cached_audio(cache_dir: Path, key: str) -> Path | None:
    for candidate in sorted(cache_dir.glob(f"{key}.*")):
        if candidate.suffix not in _SKIP_SUFFIXES and candidate.stat().st_size > 0:
            return candidate
    return None


class _QuietLogger:
    """Swallow yt-dlp's console output; errors still surface as exceptions."""

    def debug(self, msg):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


def _ydl_options(ipv4: bool | None, **extra) -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "skip_download": True,
        # yt-dlp needs a JavaScript runtime for YouTube; allow whichever is installed.
        "js_runtimes": {"deno": {}, "node": {}, "bun": {}, "quickjs": {}},
        "logger": _QuietLogger(),
    }
    # IPv4 by default. YouTube signs stream URLs for the address that asked for them; if
    # yt-dlp's requests and ffmpeg's (which section downloads use) leave over different IP
    # versions, the stream is throttled to about real time or refused (ffmpeg exit code 8).
    # ipv4=None means "let the system choose" (the fallback when IPv4 is unreachable).
    if ipv4 is not False and ipv4 is not None:
        opts["source_address"] = "0.0.0.0"
    opts.update(extra)
    return opts


def _to_video_info(raw: dict, video_id: str) -> VideoInfo:
    return VideoInfo(
        id=raw.get("id") or video_id,
        title=raw.get("title") or "(untitled)",
        channel=raw.get("channel") or raw.get("uploader") or "(unknown channel)",
        duration=float(raw["duration"]) if raw.get("duration") else None,
    )


def check_single_finished_video(raw: dict) -> None:
    if raw.get("_type") in ("playlist", "multi_video"):
        raise YouTubeError("Playlists are not supported; pass the URL of a single video.")
    if raw.get("is_live") or raw.get("live_status") in ("is_live", "is_upcoming", "post_live"):
        raise YouTubeError("Live streams and upcoming premieres are not supported; pass a finished video.")


def fetch(
    url: str,
    cache_dir: Path,
    start: float | None = None,
    end: float | None = None,
    cap: float | None = None,
    use_cache: bool = True,
    ipv4: bool = False,
    on_info=None,
) -> Download:
    """Fetch metadata and the best audio-only stream (only the needed section).

    Uses IPv4; if that fails with a network error (an IPv6-only network) and IPv4 was not
    explicitly requested, tries once more letting the system pick the address family.
    """
    try:
        return _fetch(url, cache_dir, start, end, cap, use_cache, True, on_info)
    except YouTubeError as exc:
        if ipv4 or not str(exc).startswith("Network error"):
            raise
        return _fetch(url, cache_dir, start, end, cap, use_cache, None, None)


def _fetch(url, cache_dir, start, end, cap, use_cache, ipv4, on_info) -> Download:
    import yt_dlp  # imported lazily: slow to import and only needed for URLs
    from yt_dlp.utils import download_range_func

    video_id = parse_video_id(url)
    cache_dir.mkdir(parents=True, exist_ok=True)
    info_file = cache_dir / f"{video_id}.info.json"

    raw = None
    info = None
    if use_cache and info_file.exists():
        try:
            info = VideoInfo(**json.loads(info_file.read_text(encoding="utf-8")))
        except (ValueError, TypeError):
            info = None
    try:
        if info is None:
            with yt_dlp.YoutubeDL(_ydl_options(ipv4, format="bestaudio")) as ydl:
                raw = ydl.extract_info(watch_url(video_id), download=False)
            check_single_finished_video(raw)
            info = _to_video_info(raw, video_id)
            info_file.write_text(json.dumps(asdict(info)), encoding="utf-8")

        begin, finish, capped = plan_range(start, end, info.duration, cap)
        if on_info is not None:
            on_info(info, begin, finish, capped)  # lets the tool show the title before downloading
        key = cache_key(video_id, begin, finish)
        if use_cache and (hit := cached_audio(cache_dir, key)):
            return Download(hit, info, begin, finish, capped, from_cache=True)

        options = _ydl_options(
            ipv4,
            skip_download=False,
            format="bestaudio",  # audio-only streams; never fetch video
            outtmpl={"default": str(cache_dir / f"{key}.%(ext)s")},
            overwrites=True,
        )
        if begin > 0 or finish is not None:
            # Like --download-sections: only this part is fetched.
            options["download_ranges"] = download_range_func(None, [(begin, finish if finish is not None else float("inf"))])
            options["force_keyframes_at_cuts"] = True  # exact cuts; otherwise the section can start early
        with yt_dlp.YoutubeDL(options) as ydl:
            if raw is not None:
                result = ydl.process_ie_result(raw, download=True)
            else:
                result = ydl.extract_info(watch_url(video_id), download=True)
                check_single_finished_video(result)
    except YouTubeError:
        raise
    except yt_dlp.utils.YoutubeDLError as exc:
        raise YouTubeError(explain_error(str(exc))) from None

    downloads = result.get("requested_downloads") or []
    path = Path(downloads[0]["filepath"]) if downloads and downloads[0].get("filepath") else cached_audio(cache_dir, key)
    if path is None or not path.exists():
        raise YouTubeError(f"yt-dlp reported success but no audio file was written. {UPGRADE_HINT}")
    return Download(path, info, begin, finish, capped, from_cache=False)
