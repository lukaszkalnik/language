"""A read-only choice box where a letter chooses the next value starting with it."""

from __future__ import annotations

from tkinter import ttk

from .pipeline import jump_choice


class Selector(ttk.Combobox):
    """Read-only; a choice, from the list or by a letter, fires <<ComboboxSelected>> and drops the highlight."""

    def __init__(self, master, choices: list[str], **kw) -> None:
        super().__init__(master, values=choices, state="readonly", **kw)
        self.choices = choices
        self.bind("<<ComboboxSelected>>", lambda e: self.selection_clear(), add=True)
        self.bind("<KeyPress>", lambda e: self._letter(e.char))

    def _letter(self, char: str) -> str | None:
        """The lists are long: a letter jumps to the next language starting with it."""
        found = jump_choice(self.choices, self.get(), char)
        if found is None:
            return None
        self.set(found)
        self.event_generate("<<ComboboxSelected>>")
        return "break"
