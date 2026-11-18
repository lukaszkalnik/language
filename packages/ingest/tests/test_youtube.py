import pytest
from ingest.errors import YouTubeError
from ingest.youtube import (
    check_single_finished_video,
    explain_error,
    looks_like_url,
    parse_video_id,
    plan_range,
)

VID = "QXX-uwQ-VGw"


# --- URL detection -------------------------------------------------------------

@pytest.mark.parametrize(
    "arg",
    [
        f"https://www.youtube.com/watch?v={VID}",
        f"http://youtube.com/watch?v={VID}",
        f"https://youtu.be/{VID}",
        f"youtu.be/{VID}",
        f"www.youtube.com/watch?v={VID}",
        "https://example.com/audio.mp3",
    ],
)
def test_looks_like_url(arg):
    assert looks_like_url(arg)


@pytest.mark.parametrize(
    "arg",
    ["speech.mp3", "recordings/interview.m4a", "C:\\Users\\me\\talk.wav", "./youtube.com.wav", "youtube.mp4"],
)
def test_local_paths_are_not_urls(arg):
    assert not looks_like_url(arg)


# --- video ID parsing ----------------------------------------------------------

@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VID}",
        f"https://youtube.com/watch?v={VID}&t=42s",
        f"https://www.youtube.com/watch?feature=share&v={VID}&si=abc123",
        f"https://www.youtube.com/watch?v={VID}&list=PL1234567890&index=3",
        f"https://m.youtube.com/watch?v={VID}",
        f"https://music.youtube.com/watch?v={VID}",
        f"https://youtu.be/{VID}",
        f"https://youtu.be/{VID}?si=xyz&t=10",
        f"https://www.youtube.com/shorts/{VID}",
        f"https://youtube.com/shorts/{VID}?feature=share",
        f"https://www.youtube.com/live/{VID}?si=a",
        f"https://www.youtube.com/embed/{VID}",
        f"https://www.youtube-nocookie.com/embed/{VID}",
        f"youtube.com/watch?v={VID}",
        f"  https://www.youtube.com/watch?v={VID}  ",
    ],
)
def test_parse_video_id(url):
    assert parse_video_id(url) == VID


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/playlist?list=PL590L5WQmH8fJ54F369BLDSqIwcs-TCfs",
        "https://www.youtube.com/watch?list=PL590L5WQmH8fJ54F369BLDSqIwcs-TCfs",
    ],
)
def test_playlists_are_rejected(url):
    with pytest.raises(YouTubeError, match="Playlists are not supported"):
        parse_video_id(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/@SomeChannel",
        "https://www.youtube.com/watch?v=tooshort",
        "https://youtu.be/",
        "https://www.youtube.com/",
    ],
)
def test_non_video_urls_are_rejected(url):
    with pytest.raises(YouTubeError, match="Could not find a video ID"):
        parse_video_id(url)


def test_other_sites_are_rejected():
    with pytest.raises(YouTubeError, match="Only YouTube"):
        parse_video_id("https://vimeo.com/123456")
    with pytest.raises(YouTubeError, match="Only YouTube"):
        parse_video_id("https://notyoutube.com/watch?v=" + VID)


# --- playlists and live streams found in metadata ------------------------------

def test_metadata_checks():
    check_single_finished_video({"_type": "video", "live_status": "was_live"})
    check_single_finished_video({"live_status": "not_live"})
    with pytest.raises(YouTubeError, match="Playlists"):
        check_single_finished_video({"_type": "playlist"})
    for status in ("is_live", "is_upcoming", "post_live"):
        with pytest.raises(YouTubeError, match="Live streams"):
            check_single_finished_video({"live_status": status})
    with pytest.raises(YouTubeError, match="Live streams"):
        check_single_finished_video({"is_live": True})


# --- range planning ------------------------------------------------------------

def test_default_cap_on_long_video():
    assert plan_range(None, None, duration=7200, cap=600) == (0.0, 600.0, True)


def test_no_cap_without_cap_argument():
    assert plan_range(None, None, duration=7200) == (0.0, None, False)


def test_no_cap_on_short_video():
    assert plan_range(None, None, duration=300, cap=600) == (0.0, None, False)


def test_unknown_duration_is_capped():
    assert plan_range(None, None, duration=None, cap=600) == (0.0, 600.0, True)


def test_explicit_range_is_never_capped():
    assert plan_range(60, 120, duration=7200, cap=600) == (60.0, 120.0, False)
    assert plan_range(None, 900, duration=7200, cap=600) == (0.0, 900.0, False)
    assert plan_range(30, None, duration=7200, cap=600) == (30.0, None, False)


def test_end_past_duration_means_to_the_end():
    assert plan_range(10, 999, duration=300) == (10.0, None, False)


def test_start_past_end_of_video():
    with pytest.raises(YouTubeError, match="past the end"):
        plan_range(400, None, duration=300)


# --- error messages ------------------------------------------------------------

@pytest.mark.parametrize(
    "message,expected",
    [
        ("ERROR: [youtube] abc: Private video. Sign in if you've been granted access", "private"),
        ("ERROR: [youtube] abc: Sign in to confirm your age. This video may be inappropriate for some users.", "age-restricted"),
        ("ERROR: [youtube] abc: The uploader has not made this video available in your country", "region"),
        ("ERROR: [youtube] abc: Video unavailable. This video has been removed by the uploader", "unavailable"),
        ("ERROR: [youtube] abc: This video is unavailable", "unavailable"),
        ("ERROR: [youtube] abc: This live event will begin in 3 hours.", "Live streams"),
        ("ERROR: [youtube] abc: Unable to download webpage: <urlopen error [Errno 8] nodename nor servname provided>", "Network error"),
        ("ERROR: [youtube] abc: Sign in to confirm you're not a bot", "upgrading yt-dlp"),
        ("ERROR: [youtube] abc: Some new extractor failure nobody has seen", "upgrading yt-dlp"),
    ],
)
def test_explain_error(message, expected):
    text = explain_error(message)
    assert expected in text
    assert "\n" not in text
