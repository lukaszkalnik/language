"""Text panes that fill while a step runs: pieces of output added at the end as they come."""

from __future__ import annotations

import tkinter as tk


def append(text: tk.Text, piece: str) -> None:
    """Add a piece at the end; the view follows it while it shows the end already.

    Someone who scrolled up to read stays where they are.
    """
    following = text.yview()[1] >= 0.999
    text.insert("end-1c", piece)
    if following:
        text.see("end")
