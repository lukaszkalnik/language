import socket

from ingest.cli import order_addresses, run
from ingest.errors import YouTubeError

V6 = (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2001:db8::1", 443, 0, 0))
V4 = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.1", 443))


def test_default_keeps_the_system_order():
    assert order_addresses([V6, V4]) == [V6, V4]


def test_ipv4_only():
    assert order_addresses([V6, V4], ipv4_only=True) == [V4]
    assert order_addresses([V6], ipv4_only=True) == [V6]  # nothing else to try


def test_run_turns_errors_into_one_line(capsys):
    def boom():
        raise YouTubeError("This video is private.")

    assert run(boom, "tool") == 1
    err = capsys.readouterr().err
    assert err == "tool: error: This video is private.\n"


def test_run_unexpected_error_has_no_traceback(capsys):
    def boom():
        raise RuntimeError("kaputt")

    assert run(boom, "tool") == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err and "kaputt" in err and err.count("\n") == 1


def test_ipv6_broken_only_when_ipv4_works_and_ipv6_does_not():
    import socket

    from ingest.cli import ipv6_broken

    def probe(v6, v4):
        return lambda family: v6 if family == socket.AF_INET6 else v4

    assert ipv6_broken(probe(False, True)) is True
    assert ipv6_broken(probe(True, True)) is False
    assert ipv6_broken(probe(None, True)) is False  # no IPv6 address: nothing to fix
    assert ipv6_broken(probe(False, False)) is False  # offline
    assert ipv6_broken(probe(False, None)) is False


def test_ipv6_probe_that_hangs_counts_as_broken():
    import socket
    import time

    from ingest.cli import ipv6_broken

    def probe(family):
        if family == socket.AF_INET6:
            time.sleep(1)
        return True

    t0 = time.monotonic()
    assert ipv6_broken(probe, deadline=0.2) is True
    assert time.monotonic() - t0 < 0.6


def test_choose_ipv4_env_overrides(monkeypatch):
    from ingest import cli

    monkeypatch.setattr(cli, "force_ipv4", lambda: None)
    monkeypatch.setattr(cli, "ipv6_broken", lambda: (_ for _ in ()).throw(AssertionError("probed")))
    for env, requested, expected in [("0", False, False), ("1", False, True), ("0", True, True)]:
        monkeypatch.setattr(cli, "_ip_decision", None)
        monkeypatch.setenv("LANGUAGE_IPV4", env)
        assert cli.choose_ipv4(requested) is expected
