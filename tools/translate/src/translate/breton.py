"""Breton: first into French with an M2M100 fine-tuned on Breton-French, then on from French.

Neither NLLB-200 nor the default Ollama model knows Breton well enough (NLLB has no Breton at
all; gemma4:12b mostly guesses). lgrobol's M2M100 (418M, MIT), trained on Breton-French
parallel text, translates it into French; the chosen backend takes the French on to other
languages.
"""

from __future__ import annotations

import re

from ingest.errors import ModelError
from ingest.models import hf_snapshot, pick_device, quiet_libraries

from .nllb import MAX_TOKENS, split_sentences

REPO = "lgrobol/m2m100_418M_br_fr"
REVISION = "f4b23bda90b413e25aa53e20e1d4d4947208b7fa"  # pinned: a new upload never changes results unnoticed
FILES = ["config.json", "generation_config.json", "model.safetensors", "sentencepiece.bpe.model",
         "special_tokens_map.json", "tokenizer_config.json", "vocab.json", "added_tokens.json"]
NAME = "m2m100-br-fr"
BATCH = 8
# Hesitations the Breton transcription writes out; the model, trained on written text, tries
# to translate them ("euh" became "hier matin").
FILLERS = re.compile(r"(?<!\w)(?:euh+|heu+|hum+|hm+|eh)(?!\w)[,…]*\s*", re.IGNORECASE)


def without_fillers(text: str) -> str:
    return re.sub(r"\s+([,.?!…])", r"\1", re.sub(r"\s{2,}", " ", FILLERS.sub("", text))).strip()


class BretonToFrench:
    name = NAME

    def __init__(self, device: str = "auto"):
        quiet_libraries()
        path = hf_snapshot(REPO, FILES, "model.safetensors", "M2M100 Breton-French", REVISION)
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        except Exception as exc:
            raise ModelError(f"could not load {REPO} ({exc.__class__.__name__}: {exc})") from None
        self.tokenizer.src_lang = "br"
        self.device = pick_device(device)
        self.model.to(self.device).eval()

    def _run(self, texts: list[str]) -> list[str]:
        import torch

        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                             max_length=MAX_TOKENS * 2).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**enc, forced_bos_token_id=self.tokenizer.get_lang_id("fr"),
                                      max_new_tokens=MAX_TOKENS * 2, num_beams=4)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    def __call__(self, texts: list[str], on_progress=None) -> list[str]:
        """Each text into French, sentence by sentence (the model was trained on sentences)."""
        pieces = [(i, s) for i, text in enumerate(texts) for s in split_sentences(without_fillers(text))]
        french: list[str] = []
        for start in range(0, len(pieces), BATCH):
            if on_progress:
                on_progress(start / len(pieces))
            french += self._run([s for _, s in pieces[start:start + BATCH]])
        joined = [""] * len(texts)
        for (i, _), fr in zip(pieces, french, strict=True):
            joined[i] = f"{joined[i]} {fr}".strip()
        return joined
