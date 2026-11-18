"""Offline backend: Meta's NLLB-200 through Hugging Face transformers."""

from __future__ import annotations

import re

from ingest.errors import ModelError
from ingest.models import announce_hf_download, pick_device, quiet_libraries

from . import languages
from .chunks import Chunk

MODELS = {
    "600m": "facebook/nllb-200-distilled-600M",
    "1.3b": "facebook/nllb-200-distilled-1.3B",
}
DEFAULT_MODEL = "600m"
MAX_TOKENS = 200  # NLLB was trained on sentences; longer inputs are split at sentence ends
SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text: str, max_chars: int = 400) -> list[str]:
    parts = [p for p in SENTENCE_RE.split(text.strip()) if p]
    out: list[str] = []
    for p in parts:
        if out and len(out[-1]) + len(p) < max_chars // 2:
            out[-1] += " " + p
        else:
            out.append(p)
    return out or [text]


class NLLBTranslator:
    name = "nllb"

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "auto"):
        quiet_libraries()
        repo = MODELS.get(model, model)
        announce_hf_download(repo, "pytorch_model.bin", patterns=["*.json", "*.bin", "*.model"], what=f"NLLB-200 {model}")
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(repo)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(repo)
        except Exception as exc:
            raise ModelError(f"could not load {repo} ({exc.__class__.__name__}: {exc})") from None
        self.device = pick_device(device)
        self.model.to(self.device).eval()

    def _run(self, texts: list[str], source: str, target: str) -> list[str]:
        import torch

        self.tokenizer.src_lang = languages.nllb_code(source)
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS * 2).to(self.device)
        bos = self.tokenizer.convert_tokens_to_ids(languages.nllb_code(target))
        with torch.inference_mode():
            out = self.model.generate(**enc, forced_bos_token_id=bos, max_new_tokens=MAX_TOKENS * 2, num_beams=4)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    def translate(self, chunk: Chunk, source: str | None, target: str, song: bool = False) -> list[str]:
        results = []
        for unit in chunk.units:
            src = unit.language or source
            if not src:
                raise ValueError("NLLB needs the source language: pass --from xx")
            if src == target:
                results.append(unit.text)
                continue
            sentences = split_sentences(unit.text)
            results.append(" ".join(self._run(sentences, src, target)))
        return results
