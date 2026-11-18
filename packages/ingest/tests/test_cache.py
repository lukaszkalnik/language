from pathlib import Path

from ingest import cache


def test_cache_key_includes_source_range_and_settings():
    assert cache.cache_key("QXX-uwQ-VGw", None, None) == "QXX-uwQ-VGw_0-end"
    assert cache.cache_key("QXX-uwQ-VGw", 0, None) == "QXX-uwQ-VGw_0-end"  # None start == 0
    assert cache.cache_key("QXX-uwQ-VGw", 1.5, 600) == "QXX-uwQ-VGw_1_5-600"
    assert cache.cache_key("abc", 0, 60, "vocals", "htdemucs") == "abc_0-60_vocals_htdemucs"


def test_cache_keys_differ_by_range_and_settings():
    keys = {
        cache.cache_key("a", 0, None),
        cache.cache_key("a", 0, 60),
        cache.cache_key("a", 30, 60),
        cache.cache_key("b", 0, None),
        cache.cache_key("a", 0, None, "model-x"),
    }
    assert len(keys) == 5


def test_cache_key_never_contains_path_separators():
    key = cache.cache_key("a", 0, None, "org/model")
    assert "/" not in key


def test_file_hash_is_content_based(tmp_path):
    a, b, c = tmp_path / "a.mp3", tmp_path / "b.mp3", tmp_path / "c.mp3"
    a.write_bytes(b"same bytes")
    b.write_bytes(b"same bytes")
    c.write_bytes(b"other bytes")
    assert cache.file_hash(a) == cache.file_hash(b) != cache.file_hash(c)
    assert cache.bytes_hash(b"same bytes") == cache.file_hash(a)


def test_cache_root_override(tmp_path, monkeypatch):
    monkeypatch.setenv(cache.ENV_VAR, str(tmp_path / "env"))
    cache.set_cache_root(None)
    assert cache.cache_root() == tmp_path / "env"
    cache.set_cache_root(tmp_path / "flag")  # --cache-dir wins over the variable
    try:
        assert cache.cache_dir("downloads") == tmp_path / "flag" / "downloads"
        assert (tmp_path / "flag" / "downloads").is_dir()
    finally:
        cache.set_cache_root(None)


def test_default_cache_root_is_outside_cwd(monkeypatch):
    monkeypatch.delenv(cache.ENV_VAR, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/somewhere")
    cache.set_cache_root(None)
    if cache.sys.platform != "win32":
        assert cache.cache_root() == Path("/somewhere/language")
