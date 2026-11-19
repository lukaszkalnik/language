"""Unit tests stay off the network: no IPv6/IPv4 probe at tool start (see ingest.cli.choose_ipv4)."""

import os

os.environ.setdefault("LANGUAGE_IPV4", "0")
