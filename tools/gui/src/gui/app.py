"""`language-gui`: a small window over the tools: paste a URL, pick options, read the result."""

from __future__ import annotations

import dataclasses
import queue
import re
import sys
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from . import settings, setup
from .pipeline import (
    LANGUAGES,
    Cancelled,
    InfoQuery,
    Options,
    Result,
    Runner,
    StepFailed,
    accent_possible,
    choice_code,
    choice_for,
    format_result,
    format_summary,
    format_transcript,
    format_translation,
    jump_choice,
    language_name,
    looks_like_input,
    spoken_choices,
)
from .rangeslider import RangeSlider
from .timerange import format_time

PAD = 8
LIST_ITEM = re.compile(r"^(\s*)([•–-]\s+)\S")  # leading spaces, then the bullet and its gap
INFO_DELAY_MS = 600  # wait this long after typing stops before asking for the length


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.events: queue.Queue = queue.Queue()
        self.runner: Runner | None = None
        self.result: Result | None = None
        self._job = ""  # "run" or "translate" while one is going
        self._needs_setup = setup.needed()
        self._info: InfoQuery | None = None
        self._info_url = ""
        self._info_gen = 0
        self._info_after: str | None = None
        root.title("Language")
        root.geometry("1000x700")
        root.minsize(760, 460)

        frame = ttk.Frame(root, padding=PAD)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(5, weight=1)

        ttk.Label(frame, text="URL").grid(row=0, column=0, sticky="w")
        self.url = tk.StringVar()
        entry = ttk.Entry(frame, textvariable=self.url)
        entry.grid(row=0, column=1, sticky="ew", padx=(PAD, 0))
        entry.bind("<Return>", lambda e: self.start())
        entry.focus_set()
        self.url.trace_add("write", lambda *a: self._url_changed())

        ttk.Label(frame, text="Range").grid(row=1, column=0, sticky="w", pady=(PAD, 0))
        self.slider = RangeSlider(frame, pad=PAD)
        self.slider.grid(row=1, column=1, sticky="ew", padx=(PAD, 0), pady=(PAD, 0))

        opts = ttk.Frame(frame)
        opts.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(PAD, 0))
        # Run and Stop first, so a narrow window squeezes the options rather than the buttons.
        self.stop_button = ttk.Button(opts, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="right")
        self.run_button = ttk.Button(opts, text="Run", command=self.start)
        self.run_button.pack(side="right", padx=PAD)
        self.kind = tk.StringVar(value="speech")
        ttk.Radiobutton(opts, text="Speech", value="speech", variable=self.kind).pack(side="left")
        ttk.Radiobutton(opts, text="Song", value="song", variable=self.kind).pack(side="left", padx=(PAD, 2 * PAD))
        ttk.Label(opts, text="Spoken language").pack(side="left")
        self.spoken_names = spoken_choices()
        self.spoken = ttk.Combobox(opts, values=self.spoken_names, state="readonly", width=18)
        self.spoken.set(choice_for(settings.load().get("spoken_language")))
        self.spoken.pack(side="left", padx=(4, 2 * PAD))
        self.spoken.bind("<<ComboboxSelected>>", lambda e: self._spoken_changed())
        self.spoken.bind("<KeyPress>", self._spoken_key)
        self.accent = tk.BooleanVar(value=False)
        self.accent_check = ttk.Checkbutton(opts, text="Accent (English only)", variable=self.accent)
        self.accent_check.pack(side="left")
        self._sync_accent()

        tr = ttk.Frame(frame)
        tr.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(PAD, 0))
        ttk.Label(tr, text="Translate to").pack(side="left")
        self.names = [f"{name} ({code})" for code, name in sorted(LANGUAGES.items(), key=lambda kv: kv[1])]
        self.language = ttk.Combobox(tr, values=self.names, state="readonly", width=18)
        self.language.set("English (en)")
        self.language.pack(side="left", padx=(4, PAD))
        self.translate_button = ttk.Button(tr, text="Translate", command=self.translate_now, state="disabled")
        self.translate_button.pack(side="left")
        self.translate = tk.BooleanVar(value=False)
        ttk.Checkbutton(tr, text="also right after each run", variable=self.translate).pack(side="left", padx=(PAD, 0))
        ttk.Button(tr, text="Copy all", command=self.copy_all).pack(side="right")

        busy = ttk.Frame(frame)
        busy.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(PAD, 4))
        busy.columnconfigure(0, weight=1)
        self.step_text = tk.StringVar()
        self.clock = tk.StringVar()
        bold = tkfont.nametofont("TkDefaultFont").copy()
        bold.configure(weight="bold")
        ttk.Label(busy, textvariable=self.step_text, font=bold).grid(row=0, column=0, sticky="w")
        ttk.Label(busy, textvariable=self.clock, foreground="gray").grid(row=0, column=1, sticky="e")
        self.bar = ttk.Progressbar(busy, mode="determinate", maximum=100)
        self.bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)
        self.status = tk.StringVar(value="Paste a YouTube URL (or a file path); its length appears on the "
                                         "Range slider, then press Run.")
        ttk.Label(busy, textvariable=self.status, foreground="gray").grid(row=2, column=0, sticky="w")
        self.found_language = tk.StringVar()
        ttk.Label(busy, textvariable=self.found_language).grid(row=2, column=1, sticky="e")
        self._busy = False
        self._run_started = self._step_started = 0.0
        self._last_progress = 0.0
        self._last_status = ""
        self._phase: tuple[str | None, float, float] | None = None  # label, start time, start fraction

        out = ttk.Frame(frame)
        out.grid(row=5, column=0, columnspan=2, sticky="nsew")
        out.rowconfigure(1, weight=1)
        out.columnconfigure(0, weight=1)
        ui_size = tkfont.nametofont("TkDefaultFont").cget("size")
        text_font = tkfont.nametofont("TkTextFont").copy()  # proportional, a touch larger than the UI
        text_font.configure(size=ui_size + (1 if ui_size > 0 else -1))  # negative sizes are pixels
        self.summary = tk.Text(out, wrap="word", font=text_font, height=1, undo=False, relief="flat",
                               background=self._background())
        self._make_read_only(self.summary)
        panes = ttk.PanedWindow(out, orient="horizontal")
        panes.grid(row=1, column=0, sticky="nsew")
        self.transcript_title = tk.StringVar(value="Transcript")
        self.translation_title = tk.StringVar(value="Translation")
        self.output = self._pane(panes, self.transcript_title, text_font)
        self.translation = self._pane(panes, self.translation_title, text_font)

        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self._poll)
        root.after(1000, self._tick)
        self._set_icon()
        if self._needs_setup:
            self._set_output(setup.INTRO.format(folder=setup.data_dir()))
            self.transcript_title.set("Before the first use")
            self.run_button.configure(text="Set up", command=self.start_setup)
            self.status.set("First start: the tools need to be installed once.")

    def _background(self) -> str:
        try:
            return ttk.Style().lookup("TFrame", "background") or "#ececec"
        except tk.TclError:
            return "#ececec"

    def _pane(self, panes: ttk.PanedWindow, title: tk.StringVar, font) -> tk.Text:
        box = ttk.Frame(panes)
        box.rowconfigure(1, weight=1)
        box.columnconfigure(0, weight=1)
        bold = tkfont.nametofont("TkDefaultFont").copy()
        bold.configure(weight="bold")
        ttk.Label(box, textvariable=title, font=bold).grid(row=0, column=0, sticky="w", pady=(0, 2))
        text = tk.Text(box, wrap="word", font=font, undo=False, width=40, padx=6, pady=4, spacing1=1, spacing3=3)
        yscroll = ttk.Scrollbar(box, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=yscroll.set)
        text.grid(row=1, column=0, sticky="nsew")
        yscroll.grid(row=1, column=1, sticky="ns")
        self._make_read_only(text)
        panes.add(box, weight=1)
        return text

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

    def _fill(self, text: tk.Text, value: str) -> None:
        """Show text; list items ("• ", "- ", "– ", nested by leading spaces) get a hanging indent."""
        text.delete("1.0", "end")
        text.insert("1.0", value)
        font = tkfont.Font(font=text.cget("font"))
        for number, line in enumerate(value.split("\n"), 1):
            m = LIST_ITEM.match(line)
            if m:
                tag = f"item{len(m.group(1))}"
                if tag not in text.tag_names():
                    lead = font.measure(m.group(1))
                    text.tag_configure(tag, lmargin1=lead, lmargin2=lead + font.measure(m.group(2)))
                text.tag_add(tag, f"{number}.0", f"{number}.end")

    def _set_output(self, value: str) -> None:
        """A message in the transcript pane (setup, errors); the rest is cleared."""
        self._set_summary("")
        self._fill(self.output, value)
        self._fill(self.translation, "")

    def _set_summary(self, value: str) -> None:
        self._fill(self.summary, value.rstrip())
        lines = value.strip().count("\n") + 1 if value.strip() else 0
        if lines:
            self.summary.configure(height=min(lines, 8))
            self.summary.grid(row=0, column=0, sticky="ew", pady=(0, PAD))
        else:
            self.summary.grid_remove()

    def _show_result(self, result: Result) -> None:
        self._set_summary(format_summary(result))
        self._fill(self.output, format_transcript(result))
        self._show_translation(result)

    def _show_translation(self, result: Result) -> None:
        name = language_name(result.translated_to) if result.translated_to else ""
        self.translation_title.set(f"Translation into {name}" if name else "Translation")
        hint = "" if not result.transcript.strip() else "Choose a language above and press Translate."
        self._fill(self.translation, format_translation(result) or hint)

    def _spoken_code(self) -> str | None:
        return choice_code(self.spoken.get())

    def _spoken_changed(self) -> None:
        self.spoken.selection_clear()
        settings.save({"spoken_language": self._spoken_code()})
        self._sync_accent()

    def _spoken_key(self, event) -> str | None:
        """A letter jumps to the next language starting with it (the list is long)."""
        found = jump_choice(self.spoken_names, self.spoken.get(), event.char)
        if found is None:
            return None
        self.spoken.set(found)
        self._spoken_changed()
        return "break"

    def _sync_accent(self) -> None:
        """Accent recognition is for English: greyed out while another language is chosen."""
        self.accent_check.configure(state="normal" if accent_possible(self._spoken_code()) else "disabled")

    def _language_code(self) -> str | None:
        value = self.language.get()
        return value[value.rfind("(") + 1:-1] if "(" in value else None

    def _sync_buttons(self) -> None:
        idle = self.runner is None
        self.run_button.configure(state="normal" if idle else "disabled")
        self.stop_button.configure(state="disabled" if idle else "normal")
        ready = idle and self.result is not None and bool(self.result.transcript.strip())
        self.translate_button.configure(state="normal" if ready else "disabled")

    # -- the input's length, for the range slider ------------------------------------------------
    def _url_changed(self) -> None:
        if self._info_after is not None:
            self.root.after_cancel(self._info_after)
        if self.url.get().strip() != self._info_url:
            self.slider.set_duration(None, "")
        self._info_after = self.root.after(INFO_DELAY_MS, self._fetch_info)

    def _fetch_info(self) -> None:
        self._info_after = None
        url = self.url.get().strip()
        if url == self._info_url or self._needs_setup:
            return
        if self._info:
            self._info.cancel()
            self._info = None
        self._info_url = url
        self._info_gen += 1
        if not looks_like_input(url):
            self.slider.set_duration(None, "")
            return
        self.slider.set_loading()
        query, gen = InfoQuery(), self._info_gen
        self._info = query

        def work():
            try:
                self.events.put(("info", (gen, query.run(url), None)))
            except Cancelled:
                pass
            except Exception as exc:  # noqa: BLE001 - shown next to the slider
                self.events.put(("info", (gen, None, str(exc))))

        threading.Thread(target=work, daemon=True).start()

    def _on_info(self, gen: int, info: dict | None, error: str | None) -> None:
        if gen != self._info_gen:
            return  # an answer for a URL that has been replaced since
        self._info = None
        duration = (info or {}).get("duration")
        if duration:
            self.slider.set_duration(float(duration), f"of {format_time(float(duration))}")
            if not self._busy and info.get("title"):
                self.status.set(f"{info['title']} · {format_time(float(duration))} · nothing downloaded yet: "
                                "choose a range and press Run (only that part is fetched)")
        else:
            self.slider.set_duration(None, "length unknown: whole file")
            if not self._busy:
                self.status.set(((error or "") + "; after a run the range can be set").lstrip("; ")[:200])

    # -- running ---------------------------------------------------------------------------
    def start(self) -> None:
        if self.runner is not None:
            return
        url = self.url.get().strip()
        if not url:
            self.status.set("Paste a URL first.")
            return
        start, end = self.slider.bounds() if url == self._info_url else (None, None)
        opts = Options(url=url, kind=self.kind.get(), translate_to=self._language_code() if self.translate.get() else None,
                       language=self._spoken_code(), accent=self.accent.get(), start=start, end=end)
        self.result = None
        self._set_output("")
        self.transcript_title.set("Transcript" + (f" {format_time(start or 0)}–{format_time(end) if end else 'end'}"
                                                  if start or end else ""))
        self.translation_title.set("Translation")
        self._job = "run"
        self.found_language.set("")
        self.runner = Runner(on_event=lambda kind, value: self.events.put((kind, value)))
        self._sync_buttons()
        self._start_busy()
        runner = self.runner

        def work():
            try:
                self.events.put(("done", runner.run(opts)))
            except Cancelled:
                self.events.put(("stopped", None))
            except StepFailed as exc:
                self.events.put(("error", str(exc)))
            except Exception as exc:  # noqa: BLE001 - shown in the window, not a traceback
                self.events.put(("error", f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=work, daemon=True).start()

    def translate_now(self) -> None:
        """Translate the transcript already shown; a new language replaces the translation."""
        to = self._language_code()
        if self.runner is not None or self.result is None or not to:
            return
        result = dataclasses.replace(self.result)
        self._job = "translate"
        self.runner = Runner(on_event=lambda kind, value: self.events.put((kind, value)))
        runner = self.runner
        self._sync_buttons()
        self._start_busy()
        self.translation_title.set(f"Translation into {language_name(to)}")
        self._fill(self.translation, "")

        def work():
            try:
                self.events.put(("translated", runner.translate(result, to)))
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
        if self.result is not None:
            text = format_result(self.result)
        else:
            text = "\n\n".join(t for t in (self.output.get("1.0", "end-1c").strip(),
                                            self.translation.get("1.0", "end-1c").strip()) if t) + "\n"
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status.set("Copied to the clipboard.")

    def close(self) -> None:
        if self.runner:
            self.runner.cancel()
        if self._info:
            self._info.cancel()
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
        self.bar.configure(mode="determinate", value=100 if kind in ("done", "translated", "setup-done") else 0)
        took = _clock(time.monotonic() - self._run_started)
        self.step_text.set({"done": "Finished", "translated": "Translated", "setup-done": "Set up finished",
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
                if kind == "info":
                    self._on_info(*value)
                    continue
                if kind == "duration":
                    # --info gave no length: the run did, so the next run can use a range
                    if self.slider.range is None and self.url.get().strip() == self._info_url:
                        self.slider.set_duration(value, f"of {format_time(value)}")
                    continue
                self._stop_busy(kind)
                if kind == "setup-done":
                    self._needs_setup = False
                    self.transcript_title.set("Transcript")
                    self.run_button.configure(text="Run", command=self.start, state="normal")
                    self._set_output("Ready. Paste a URL and press Run.\n\nThe first run of each feature "
                                     "downloads its model; the status line shows the progress.\n")
                    self.status.set("Setup finished.")
                    self._info_url = ""
                    self._fetch_info()
                    continue
                if kind == "setup-failed":
                    self.run_button.configure(state="normal")
                    self._set_output(f"Setup failed: {value}\n\nCheck the internet connection and "
                                     "press “Set up” to try again; finished parts are kept.\n")
                    self.status.set("Setup failed.")
                    continue
                job, self._job = self._job, ""
                self.runner = None
                if kind == "done":
                    self.result = value
                    self._show_result(value)
                    self.status.set("Done.")
                elif kind == "translated":
                    self.result = value
                    self._show_translation(value)
                    self.status.set(f"Translated into {language_name(value.translated_to)}.")
                elif kind == "stopped":
                    self.status.set("Stopped.")
                    if job == "translate" and self.result is not None:
                        self._show_translation(self.result)
                elif job == "translate":
                    self._fill(self.translation, f"error: {value}\n")  # the transcript stays
                    self.status.set("Translation failed.")
                else:
                    self._set_output(f"error: {value}\n")
                    self.status.set("Failed.")
                self._sync_buttons()
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
