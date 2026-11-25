"""Conventions shared by every tool's command line: input arguments, stderr status, exit codes."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import sys
import threading
import time
from collections.abc import Callable

from . import cache
from .errors import UserError
from .timeparse import parse_time


def status(message: str = "") -> None:
    """Anything for humans goes to stderr, so stdout stays clean for pipes."""
    print(message, file=sys.stderr, flush=True)


PROGRESS_ENV = "LANGUAGE_PROGRESS"
_last_progress: dict[str, float] = {}


def progress_wanted() -> bool:
    """Whether a front end (the desktop window) asked for machine-readable progress lines."""
    return os.environ.get(PROGRESS_ENV) == "1"


def report_progress(label: str, fraction: float, every: float = 0.5) -> None:
    """With LANGUAGE_PROGRESS=1, print `@progress 0.42 Label` on stderr (at most every `every` s).

    Terminal output is unchanged without the variable; a front end turns these lines into a
    progress bar instead of showing them.
    """
    if not progress_wanted():
        return
    now = time.monotonic()
    fraction = min(1.0, max(0.0, fraction))
    if fraction < 1 and now - _last_progress.get(label, -every) < every:
        return
    _last_progress[label] = now
    print(f"@progress {fraction:.3f} {label}", file=sys.stderr, flush=True)


def report_partial(text: str) -> None:
    """With LANGUAGE_PROGRESS=1, print `@partial "<text as JSON>"` on stderr: the next piece of output.

    A front end shows the pieces as they come; the result on stdout, once done, replaces them
    (it may differ: the tools tidy up the whole text at the end). ASCII-only JSON, so any
    stderr encoding takes it.
    """
    if text and progress_wanted():
        print(f"@partial {json.dumps(text)}", file=sys.stderr, flush=True)


def time_arg(value: str) -> float:
    try:
        return parse_time(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="audio/video file (anything ffmpeg reads), a YouTube URL, or - for a WAV stream on stdin")
    parser.add_argument("--start", type=time_arg, metavar="TIME", help="start here (seconds, mm:ss or hh:mm:ss)")
    parser.add_argument("--end", type=time_arg, metavar="TIME", help="stop here (seconds, mm:ss or hh:mm:ss)")
    parser.add_argument("--no-cache", action="store_true", help="ignore cached downloads and results; compute again")
    parser.add_argument("--cache-dir", metavar="DIR", help=f"shared cache folder (default: {cache.cache_root()}; or set {cache.ENV_VAR})")
    parser.add_argument("--ipv4", action="store_true", help="use IPv4 only for downloads, e.g. on a network with broken IPv6 (or set LANGUAGE_IPV4=1)")
    parser.add_argument("--info", action="store_true",
                        help="only print the input's title and duration as JSON and exit (no download, no model)")
    parser.set_defaults(info_prog=parser.prog)


def order_addresses(results: list, ipv4_only: bool = False) -> list:
    """Keep only IPv4 addresses when asked to (and some exist); otherwise leave the order alone."""
    if not ipv4_only:
        return results
    v4 = [r for r in results if r[0] == socket.AF_INET]
    return v4 or results


_original_getaddrinfo = socket.getaddrinfo


def force_ipv4() -> None:
    """Make every library in this process resolve hostnames to IPv4 only (--ipv4).

    Python has no "happy eyeballs": on a network whose IPv6 is broken, a connection to an
    IPv6 address hangs until a long timeout instead of falling back to IPv4.
    """
    def patched(*args, **kwargs):
        return order_addresses(_original_getaddrinfo(*args, **kwargs), True)

    socket.getaddrinfo = patched


PROBE_HOST = "huggingface.co"  # where models come from; also reachable over both IP versions
PROBE_TIMEOUT = 1.5


def can_connect(family: int, host: str = PROBE_HOST, port: int = 443, timeout: float = PROBE_TIMEOUT) -> bool | None:
    """Whether a TCP connection over this IP version works; None if the host has no such address."""
    try:
        infos = _original_getaddrinfo(host, port, family, socket.SOCK_STREAM)
    except OSError:
        return None
    if not infos:
        return None
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(infos[0][-1])
            return True
    except OSError:
        return False


def ipv6_broken(probe: Callable[[int], bool | None] = can_connect, deadline: float = PROBE_TIMEOUT + 0.3) -> bool:
    """True when IPv4 connects but IPv6 does not (e.g. a router advertising a stale IPv6 prefix).

    Both are tried at once, so a working network pays one round trip and a broken one about
    PROBE_TIMEOUT seconds. Offline, or with no IPv6 address at all, nothing is changed.
    """
    results: dict[int, bool | None] = {}

    def run_probe(family):
        results[family] = probe(family)

    threads = [threading.Thread(target=run_probe, args=(f,), daemon=True) for f in (socket.AF_INET6, socket.AF_INET)]
    for t in threads:
        t.start()
    end = time.monotonic() + deadline
    for t in threads:
        t.join(max(0.0, end - time.monotonic()))
    return results.get(socket.AF_INET) is True and results.get(socket.AF_INET6, False) is False


_ip_decision: bool | None = None


def choose_ipv4(requested: bool = False) -> bool:
    """Use IPv4 only when asked (--ipv4, LANGUAGE_IPV4=1) or when IPv6 turns out to be broken.

    LANGUAGE_IPV4=0 skips the check. Decided once per process; forces IPv4 for every library.
    """
    global _ip_decision
    if _ip_decision is None:
        env = os.environ.get("LANGUAGE_IPV4")
        if requested or env == "1":
            _ip_decision = True
        elif env == "0":
            _ip_decision = False
        else:
            _ip_decision = ipv6_broken()
            if _ip_decision:
                status("IPv6 is not working on this network; using IPv4 (LANGUAGE_IPV4=0 to turn this off).")
        if _ip_decision:
            force_ipv4()
    return _ip_decision


def apply_common(args: argparse.Namespace) -> None:
    if getattr(args, "cache_dir", None):
        cache.set_cache_root(args.cache_dir)
    args.ipv4 = choose_ipv4(getattr(args, "ipv4", False))
    if getattr(args, "info", False):
        sys.exit(run(lambda: print_info(args), getattr(args, "info_prog", "language")))


def print_info(args: argparse.Namespace) -> None:
    """--info: the input's title and duration as one JSON line on stdout."""
    from .source import input_info  # here: source imports youtube, which imports this module

    info = input_info(args.input, use_cache=not getattr(args, "no_cache", False), ipv4=args.ipv4)
    print(json.dumps(info, ensure_ascii=False), flush=True)


def run(main: Callable[[], int | None], prog: str) -> int:
    """Run a tool's main: one `error:` line and a non-zero exit code instead of a traceback."""
    debug = os.environ.get("LANGUAGE_DEBUG") == "1"
    try:
        return main() or 0
    except UserError as exc:
        print(f"{prog}: error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        if debug:
            raise
        print(f"{prog}: error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(f"{prog}: interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        # The reader of our stdout went away (e.g. `| head`); not an error worth a message.
        with contextlib.suppress(OSError):
            sys.stdout = open(os.devnull, "w")  # noqa: SIM115 - silence the flush at exit
        return 141
    except Exception as exc:  # last resort
        if debug:
            raise
        print(f"{prog}: error: unexpected {exc.__class__.__name__}: {exc} (set LANGUAGE_DEBUG=1 for a traceback)", file=sys.stderr)
        return 1
