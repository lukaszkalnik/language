"""`language-gui`: a small window over the tools: paste a URL, pick options, read the result."""

from __future__ import annotations

import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from . import setup
from .pipeline import LANGUAGES, Cancelled, Options, Runner, StepFailed, format_result

PAD = 8


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.events: queue.Queue = queue.Queue()
        self.runner: Runner | None = None
        root.title("Language")
        root.geometry("900x640")
        root.minsize(560, 400)

        frame = ttk.Frame(root, padding=PAD)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(4, weight=1)

        ttk.Label(frame, text="URL").grid(row=0, column=0, sticky="w")
        self.url = tk.StringVar()
        entry = ttk.Entry(frame, textvariable=self.url)
        entry.grid(row=0, column=1, sticky="ew", padx=(PAD, 0))
        entry.bind("<Return>", lambda e: self.start())
        entry.focus_set()

        opts = ttk.Frame(frame)
        opts.grid(row=1, column=0, columnspan=2, sticky="w", pady=PAD)
        self.kind = tk.StringVar(value="speech")
        ttk.Radiobutton(opts, text="Speech", value="speech", variable=self.kind).pack(side="left")
        ttk.Radiobutton(opts, text="Song", value="song", variable=self.kind).pack(side="left", padx=(PAD, 3 * PAD))

        self.translate = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Translate to", variable=self.translate,
                        command=self._sync_language).pack(side="left")
        self.names = [f"{name} ({code})" for code, name in sorted(LANGUAGES.items(), key=lambda kv: kv[1])]
        self.language = ttk.Combobox(opts, values=self.names, state="disabled", width=18)
        self.language.set("English (en)")
        self.language.pack(side="left", padx=(4, 3 * PAD))

        self.accent = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Accent recognition (English only)", variable=self.accent).pack(side="left")

        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, columnspan=2, sticky="w")
        self.run_button = ttk.Button(buttons, text="Run", command=self.start)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=PAD)
        ttk.Button(buttons, text="Copy all", command=self.copy_all).pack(side="left")

        busy = ttk.Frame(frame)
        busy.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(PAD, 4))
        busy.columnconfigure(0, weight=1)
        self.step_text = tk.StringVar()
        self.clock = tk.StringVar()
        bold = tkfont.nametofont("TkDefaultFont").copy()
        bold.configure(weight="bold")
        ttk.Label(busy, textvariable=self.step_text, font=bold).grid(row=0, column=0, sticky="w")
        ttk.Label(busy, textvariable=self.clock, foreground="gray").grid(row=0, column=1, sticky="e")
        self.bar = ttk.Progressbar(busy, mode="determinate", maximum=100)
        self.bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)
        self.status = tk.StringVar(value="Paste a YouTube URL (or a file path) and press Run.")
        ttk.Label(busy, textvariable=self.status, foreground="gray").grid(row=2, column=0, sticky="w")
        self.found_language = tk.StringVar()
        ttk.Label(busy, textvariable=self.found_language).grid(row=2, column=1, sticky="e")
        self._busy = False
        self._run_started = self._step_started = 0.0
        self._last_progress = 0.0
        self._last_status = ""
        self._phase: tuple[str | None, float, float] | None = None  # label, start time, start fraction

        out = ttk.Frame(frame)
        out.grid(row=4, column=0, columnspan=2, sticky="nsew")
        out.rowconfigure(0, weight=1)
        out.columnconfigure(0, weight=1)
        mono = tkfont.nametofont("TkFixedFont").copy()
        mono.configure(size=max(mono.cget("size"), 12))
        self.output = tk.Text(out, wrap="none", font=mono, undo=False)
        yscroll = ttk.Scrollbar(out, orient="vertical", command=self.output.yview)
        xscroll = ttk.Scrollbar(out, orient="horizontal", command=self.output.xview)
        self.output.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.output.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self._make_read_only(self.output)

        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self._poll)
        root.after(1000, self._tick)
        self._set_icon()
        if setup.needed():
            self._set_output(setup.INTRO.format(folder=setup.data_dir()))
            self.run_button.configure(text="Set up", command=self.start_setup)
            self.status.set("First start: the tools need to be installed once.")

    def _set_icon(self) -> None:
        icon = setup.bundle_dir() / "icon.png"
        if icon.is_file():
            try:
                self._icon = tk.PhotoImage(file=str(icon))
                self.root.iconphoto(True, self._icon)
            except tk.TclError:
                pass

    def start_setup(self) -> None:
        self.run_button.configure(state="disabled")
        self._start_busy()
        self._set_output(setup.INTRO.format(folder=setup.data_dir()).rsplit("Press", 1)[0]
                         + "Installing... the status line below shows progress.\n")

        def work():
            try:
                setup.run_setup(lambda s: self.events.put(("status", s)),
                                lambda i, n, title: self.events.put(("step", (i, n, title))))
                self.events.put(("setup-done", None))
            except Exception as exc:  # noqa: BLE001 - shown in the window
                self.events.put(("setup-failed", str(exc)))

        threading.Thread(target=work, daemon=True).start()

    # -- read-only text: selectable and copyable, but not editable --------------------------
    def _make_read_only(self, text: tk.Text) -> None:
        mod = "Command" if sys.platform == "darwin" else "Control"
        allowed = {"c", "a", "C", "A"}

        def key(event):
            ctrl = event.state & (0x4 | 0x8 | 0x10)  # Control / Mod1 / Command
            if ctrl and event.keysym in allowed:
                return None
            if event.keysym in ("Left", "Right", "Up", "Down", "Home", "End", "Prior", "Next"):
                return None
            return "break"

        text.bind("<Key>", key)
        for seq in ("<<Paste>>", "<<Cut>>", "<<Clear>>", "<<PasteSelection>>", "<Button-2>"):
            text.bind(seq, lambda e: "break")
        text.bind(f"<{mod}-a>", lambda e: (text.tag_add("sel", "1.0", "end-1c"), "break")[1])

        menu = tk.Menu(text, tearoff=False)
        menu.add_command(label="Copy", command=lambda: text.event_generate("<<Copy>>"))
        menu.add_command(label="Select all", command=lambda: text.tag_add("sel", "1.0", "end-1c"))
        right = "<Button-2>" if sys.platform == "darwin" else "<Button-3>"
        text.bind(right, lambda e: (menu.tk_popup(e.x_root, e.y_root), "break")[1])
        if sys.platform == "darwin":
            text.bind("<Control-Button-1>", lambda e: (menu.tk_popup(e.x_root, e.y_root), "break")[1])

    def _set_output(self, value: str) -> None:
        self.output.delete("1.0", "end")
        self.output.insert("1.0", value)

    def _sync_language(self) -> None:
        self.language.configure(state="readonly" if self.translate.get() else "disabled")

    def _language_code(self) -> str | None:
        if not self.translate.get():
            return None
        value = self.language.get()
        return value[value.rfind("(") + 1:-1] if "(" in value else None

    # -- running ---------------------------------------------------------------------------
    def start(self) -> None:
        if self.runner is not None:
            return
        url = self.url.get().strip()
        if not url:
            self.status.set("Paste a URL first.")
            return
        opts = Options(url=url, kind=self.kind.get(), translate_to=self._language_code(),
                       accent=self.accent.get())
        self._set_output("")
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._start_busy()
        self.found_language.set("")
        self.runner = Runner(on_event=lambda kind, value: self.events.put((kind, value)))
        runner = self.runner

        def work():
            try:
                self.events.put(("done", format_result(runner.run(opts))))
            except Cancelled:
                self.events.put(("stopped", None))
            except StepFailed as exc:
                self.events.put(("error", str(exc)))
            except Exception as exc:  # noqa: BLE001 - shown in the window, not a traceback
                self.events.put(("error", f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=work, daemon=True).start()

    def stop(self) -> None:
        if self.runner:
            self.runner.cancel()
            self.status.set("Stopping...")

    def copy_all(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self.output.get("1.0", "end-1c"))
        self.status.set("Copied to the clipboard.")

    def close(self) -> None:
        if self.runner:
            self.runner.cancel()
        self.root.destroy()

    # -- progress: step, bar (determinate with a percentage, animated without), clock ----------
    def _start_busy(self) -> None:
        self._busy = True
        self._run_started = self._step_started = time.monotonic()
        self.step_text.set("Starting...")
        self.clock.set("0:00")
        self._indeterminate()

    def _stop_busy(self, kind: str) -> None:
        self._busy = False
        self.bar.stop()
        self.bar.configure(mode="determinate", value=100 if kind in ("done", "setup-done") else 0)
        took = _clock(time.monotonic() - self._run_started)
        self.step_text.set({"done": "Finished", "setup-done": "Set up finished",
                            "stopped": "Stopped"}.get(kind, "Failed"))
        self.clock.set(f"took {took}")

    def _indeterminate(self) -> None:
        self._phase = None
        if str(self.bar.cget("mode")) != "indeterminate":
            self.bar.configure(mode="indeterminate", value=0)
            self.bar.start(15)

    def _on_step(self, index: int, count: int, title: str) -> None:
        self.step_text.set(f"Step {index} of {count} · {title}")
        self._step_started = time.monotonic()
        self._last_status = ""
        self.status.set(f"{title}...")
        self._indeterminate()

    def _on_status(self, text: str) -> None:
        self._last_status = text
        self.status.set(text[:200])
        # a new message starts a new phase, unless it is a progress line's text twin
        if time.monotonic() - self._last_progress > 3:
            self._indeterminate()

    def _on_progress(self, fraction: float, label: str | None) -> None:
        now = time.monotonic()
        self._last_progress = now
        if str(self.bar.cget("mode")) != "determinate":
            self.bar.stop()
            self.bar.configure(mode="determinate")
        if self._phase is None or self._phase[0] != label or fraction < self._phase[2]:
            self._phase = (label, now, fraction)
        self.bar.configure(value=fraction * 100)
        base = label or self._last_status.rstrip(". ") or "Working"
        text = f"{base} — {fraction:.0%}"
        _, t0, f0 = self._phase
        if now - t0 > 5 and fraction - f0 > 0.02 and fraction < 1:
            left = (now - t0) / (fraction - f0) * (1 - fraction)
            text += f" · about {_clock(left)} left"
        self.status.set(text[:200])

    def _tick(self) -> None:
        if self._busy:
            now = time.monotonic()
            step = _clock(now - self._step_started)
            total = _clock(now - self._run_started)
            self.clock.set(step if step == total else f"{step} · total {total}")
        self.root.after(1000, self._tick)

    def _poll(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "status":
                    self._on_status(value)
                    continue
                if kind == "step":
                    self._on_step(*value)
                    continue
                if kind == "language":
                    self.found_language.set(f"Language: {value}")
                    continue
                if kind == "progress":
                    self._on_progress(*value)
                    continue
                self._stop_busy(kind)
                if kind == "setup-done":
                    self.run_button.configure(text="Run", command=self.start, state="normal")
                    self._set_output("Ready. Paste a URL and press Run.\n\nThe first run of each feature "
                                     "downloads its model; the status line shows the progress.\n")
                    self.status.set("Setup finished.")
                    continue
                if kind == "setup-failed":
                    self.run_button.configure(state="normal")
                    self._set_output(f"Setup failed: {value}\n\nCheck the internet connection and "
                                     "press “Set up” to try again; finished parts are kept.\n")
                    self.status.set("Setup failed.")
                    continue
                self.runner = None
                self.run_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                if kind == "done":
                    self._set_output(value)
                    self.status.set("Done.")
                elif kind == "stopped":
                    self.status.set("Stopped.")
                else:
                    self._set_output(f"error: {value}\n")
                    self.status.set("Failed.")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)


def _clock(seconds: float) -> str:
    s = int(max(0, seconds))
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def main() -> int:
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
