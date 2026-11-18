import socket

from ingest.cli import order_addresses, run
from ingest.errors import YouTubeError

V6 = (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2001:db8::1", 443, 0, 0))
V4 = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.1", 443))


def test_ipv4_first_keeps_ipv6_fallback():
    assert order_addresses([V6, V4]) == [V4, V6]
    assert order_addresses([V6]) == [V6]


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
