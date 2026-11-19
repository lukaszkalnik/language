"""A two-handle time slider on a Tk Canvas, with start/end fields kept in sync (Tk has none).

Drag a handle, click the track to move the nearer handle there, or use the arrow keys on the
focused slider (Shift: 10 s; Tab between the fields). The values live in a `TimeRange`.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from .timerange import TimeRange, format_time

TRACK = "#c9c9c9"
SPAN = "#3b82c4"
DISABLED = "#e2e2e2"
HANDLE_FILL = "#ffffff"
HANDLE_ACTIVE = "#dceafa"
R = 7  # handle radius
HEIGHT = 2 * R + 8


class RangeSlider(ttk.Frame):
    def __init__(self, master, on_change: Callable[[], None] | None = None, pad: int = 8):
        super().__init__(master)
        self.on_change = on_change or (lambda: None)
        self.range: TimeRange | None = None
        self.active = "start"
        self._drag: str | None = None
        self.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, height=HEIGHT, highlightthickness=1, takefocus=1, bd=0,
                                highlightbackground=self._background(), highlightcolor=SPAN,
                                background=self._background())
        self.canvas.grid(row=0, column=0, sticky="ew")
        self.start_text = tk.StringVar(value="0:00")
        self.end_text = tk.StringVar(value="0:00")
        self.start_entry = ttk.Entry(self, textvariable=self.start_text, width=8, justify="center")
        self.start_entry.grid(row=0, column=1, padx=(pad, 2))
        ttk.Label(self, text="–").grid(row=0, column=2)
        self.end_entry = ttk.Entry(self, textvariable=self.end_text, width=8, justify="center")
        self.end_entry.grid(row=0, column=3, padx=(2, pad))
        self.note = tk.StringVar()
        ttk.Label(self, textvariable=self.note, foreground="gray", width=22).grid(row=0, column=4, sticky="w")
        self.loading = ttk.Progressbar(self, mode="indeterminate", length=70)

        for entry, which in ((self.start_entry, "start"), (self.end_entry, "end")):
            entry.bind("<Return>", lambda e, w=which: self._typed(w))
            entry.bind("<KP_Enter>", lambda e, w=which: self._typed(w))
            entry.bind("<FocusOut>", lambda e, w=which: self._typed(w))
            entry.bind("<Escape>", lambda e: self._show_values())
        c = self.canvas
        c.bind("<Configure>", lambda e: self._draw())
        c.bind("<ButtonPress-1>", self._press)
        c.bind("<B1-Motion>", self._motion)
        c.bind("<ButtonRelease-1>", self._release)
        for key, step in (("Left", -1), ("Right", 1), ("Shift-Left", -10), ("Shift-Right", 10)):
            c.bind(f"<{key}>", lambda e, s=step: self._nudge(s))
        c.bind("<Home>", lambda e: self._move(self.active, 0))
        c.bind("<End>", lambda e: self._move(self.active, self.range.duration if self.range else 0))
        for key in ("<Up>", "<Down>"):  # switch handles
            c.bind(key, lambda e: self._switch())
        c.bind("<FocusIn>", lambda e: self._draw())
        c.bind("<FocusOut>", lambda e: self._draw())
        self.set_duration(None, "")

    def _background(self) -> str:
        try:
            return ttk.Style().lookup("TFrame", "background") or "#ececec"
        except tk.TclError:
            return "#ececec"

    # -- state -------------------------------------------------------------------------------
    def set_loading(self, message: str = "Reading the length…") -> None:
        self.range = None
        self.note.set(message)
        self.loading.grid(row=0, column=5, padx=(4, 0))
        self.loading.start(12)
        self._enable(False)

    def set_duration(self, duration: float | None, message: str | None = None) -> None:
        """A known duration enables the slider over the whole file; None disables it."""
        self.loading.stop()
        self.loading.grid_remove()
        self.range = TimeRange(duration) if duration else None
        if self.range:
            self.note.set(message if message is not None else f"of {format_time(duration)}")
        else:
            self.note.set(message if message is not None else "whole file")
        self._enable(self.range is not None)
        self._show_values()
        self.on_change()

    def bounds(self) -> tuple[float | None, float | None]:
        """(start, end) to pass to the tools; (None, None) for the whole file or no slider."""
        return self.range.bounds() if self.range else (None, None)

    def _enable(self, on: bool) -> None:
        state = "normal" if on else "disabled"
        self.start_entry.configure(state=state)
        self.end_entry.configure(state=state)
        self.canvas.configure(takefocus=1 if on else 0)
        self._draw()

    def _show_values(self) -> None:
        if self.range:
            self.start_text.set(self.range.label("start"))
            self.end_text.set(self.range.label("end"))
        else:
            self.start_text.set("")
            self.end_text.set("")
        self._draw()

    # -- drawing -----------------------------------------------------------------------------
    def _x(self, seconds: float) -> float:
        width = max(self.canvas.winfo_width(), 4 * R)
        span = self.range.duration if self.range else 1
        return R + 2 + (width - 2 * R - 4) * (seconds / span)

    def _seconds(self, x: float) -> float:
        width = max(self.canvas.winfo_width(), 4 * R)
        frac = (x - R - 2) / max(1, width - 2 * R - 4)
        return min(1.0, max(0.0, frac)) * (self.range.duration if self.range else 0)

    def _draw(self) -> None:
        c = self.canvas
        c.delete("all")
        y = HEIGHT / 2
        left, right = R + 2, max(c.winfo_width(), 4 * R) - R - 2
        on = self.range is not None
        c.create_line(left, y, right, y, fill=TRACK if on else DISABLED, width=4, capstyle="round")
        if not on:
            return
        x0, x1 = self._x(self.range.start), self._x(self.range.end)
        c.create_line(x0, y, x1, y, fill=SPAN, width=4)
        focused = self.focus_get() is c
        for which, x in (("start", x0), ("end", x1)):
            fill = HANDLE_ACTIVE if focused and which == self.active else HANDLE_FILL
            c.create_oval(x - R, y - R, x + R, y + R, fill=fill, outline=SPAN, width=2)

    # -- interaction ---------------------------------------------------------------------------
    def _move(self, which: str, seconds: float) -> None:
        if not self.range:
            return
        before = (self.range.start, self.range.end)
        if which == "start":
            self.range.set_start(seconds)
        else:
            self.range.set_end(seconds)
        self.active = which
        self._show_values()
        if (self.range.start, self.range.end) != before:
            self.on_change()

    def _nearest(self, x: float) -> str:
        x0, x1 = self._x(self.range.start), self._x(self.range.end)
        if abs(x - x0) == abs(x - x1):  # handles on top of each other: go the way of the click
            return "start" if x < x0 else "end"
        return "start" if abs(x - x0) < abs(x - x1) else "end"

    def _press(self, event) -> None:
        if not self.range:
            return
        self.canvas.focus_set()
        self._drag = self._nearest(event.x)
        self._move(self._drag, self._seconds(event.x))

    def _motion(self, event) -> None:
        if self._drag:
            self._move(self._drag, self._seconds(event.x))

    def _release(self, event) -> None:
        self._drag = None

    def _nudge(self, step: int):
        if self.range:
            self._move(self.active, (self.range.start if self.active == "start" else self.range.end) + step)
        return "break"

    def _switch(self):
        self.active = "end" if self.active == "start" else "start"
        self._draw()
        return "break"

    def _typed(self, which: str) -> None:
        if not self.range:
            return
        text = (self.start_text if which == "start" else self.end_text).get()
        before = (self.range.start, self.range.end)
        try:
            self.range.set_text(which, text)
        except ValueError:
            self.note.set("type m:ss or h:mm:ss")
            self.after(2500, lambda: self.note.set(f"of {format_time(self.range.duration)}") if self.range else None)
        self.active = which
        self._show_values()
        if (self.range.start, self.range.end) != before:
            self.on_change()
