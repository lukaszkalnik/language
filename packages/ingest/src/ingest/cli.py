"""Conventions shared by every tool's command line: input arguments, stderr status, exit codes."""

from __future__ import annotations

import argparse
import os
import socket
import sys
from collections.abc import Callable

from . import cache
from .errors import UserError
from .timeparse import parse_time


def status(message: str = "") -> None:
    """Anything for humans goes to stderr, so stdout stays clean for pipes."""
    print(message, file=sys.stderr, flush=True)


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
    parser.add_argument("--ipv4", action="store_true", help="use IPv4 only for downloads (IPv4 is always tried first; or set LANGUAGE_IPV4=1)")


def order_addresses(results: list, ipv4_only: bool = False) -> list:
    """IPv4 addresses first (or only), so a network with broken IPv6 does not hang downloads.

    Python tries addresses in order and has no "happy eyeballs", so an IPv6 address that
    never answers blocks until a long timeout. IPv6 stays as a fallback unless ipv4_only.
    """
    v4 = [r for r in results if r[0] == socket.AF_INET]
    if ipv4_only:
        return v4 or results
    return v4 + [r for r in results if r[0] != socket.AF_INET]


_original_getaddrinfo = socket.getaddrinfo


def prefer_ipv4(strict: bool = False) -> None:
    def patched(*args, **kwargs):
        return order_addresses(_original_getaddrinfo(*args, **kwargs), strict)

    socket.getaddrinfo = patched


def apply_common(args: argparse.Namespace) -> None:
    if getattr(args, "cache_dir", None):
        cache.set_cache_root(args.cache_dir)
    args.ipv4 = getattr(args, "ipv4", False) or os.environ.get("LANGUAGE_IPV4") == "1"
    prefer_ipv4(strict=args.ipv4)


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
