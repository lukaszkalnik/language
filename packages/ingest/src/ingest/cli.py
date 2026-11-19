"""Conventions shared by every tool's command line: input arguments, stderr status, exit codes."""

from __future__ import annotations

import argparse
import os
import socket
import sys
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


def apply_common(args: argparse.Namespace) -> None:
    if getattr(args, "cache_dir", None):
        cache.set_cache_root(args.cache_dir)
    args.ipv4 = getattr(args, "ipv4", False) or os.environ.get("LANGUAGE_IPV4") == "1"
    if args.ipv4:
        force_ipv4()


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
        try:
            sys.stdout = open(os.devnull, "w")  # noqa: SIM115 - silence the flush at exit
        except OSError:
            pass
        return 141
    except Exception as exc:  # last resort
        if debug:
            raise
        print(f"{prog}: error: unexpected {exc.__class__.__name__}: {exc} (set LANGUAGE_DEBUG=1 for a traceback)", file=sys.stderr)
        return 1
