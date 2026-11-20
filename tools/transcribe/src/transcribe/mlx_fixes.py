"""Fixes for mlx-whisper 0.4.3, applied when the MLX backend loads.

Its timestamp rules differ from OpenAI's reference decoder in two ways:
- "timestamps must not go back" compares the *position* of the last timestamp in the
  sequence instead of its value, so earlier times stay allowed;
- "if timestamps are likelier than any word, write a timestamp" is decided on the logits
  *before* the other rules are applied. When the model wants a timestamp where none is
  allowed (right after the first one, say), every token is then forbidden: decoding picks
  token 0 ("!") and the segment's log-probability becomes NaN, which disables the
  temperature fallback and the hallucination checks, and breaks the timestamps.

Fine-tuned models such as KB-Whisper run into this all the time; large-v3 now and then.
"""

from __future__ import annotations

import numpy as np


def apply_timestamp_rules(self, logits, tokens):
    """ApplyTimestampRules.apply as in openai-whisper's decoding.py."""
    import mlx.core as mx

    tok = self.tokenizer
    begin = tok.timestamp_begin
    mask = np.zeros(logits.shape, np.float32)
    if tok.no_timestamps is not None:
        mask[:, tok.no_timestamps] = -np.inf  # handled by without_timestamps
    rows = tokens.tolist()
    for k, row in enumerate(rows):
        seq = row[self.sample_begin:]
        last_was_timestamp = len(seq) >= 1 and seq[-1] >= begin
        penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= begin
        # timestamps come in pairs, except directly before the end
        if last_was_timestamp:
            if penultimate_was_timestamp:
                mask[k, begin:] = -np.inf  # has to be text
            else:
                mask[k, : tok.eot] = -np.inf  # cannot be text
        stamps = [t for t in seq if t >= begin]
        if stamps:
            # timestamps must not decrease, and a segment must not be empty (no endless loops)
            last = stamps[-1] if last_was_timestamp and not penultimate_was_timestamp else stamps[-1] + 1
            mask[k, begin:last] = -np.inf
    if len(rows[0]) == self.sample_begin:
        mask[:, :begin] = -np.inf  # the first token is a timestamp
        if self.max_initial_timestamp_index is not None:
            mask[:, begin + self.max_initial_timestamp_index + 1:] = -np.inf

    logits = logits + mx.array(mask)
    # if all timestamps together are likelier than any single text token, write a timestamp
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    timestamp_logprob = mx.logsumexp(logprobs[:, begin:], axis=-1, keepdims=True)
    max_text_logprob = mx.max(logprobs[:, :begin], axis=-1, keepdims=True)
    columns = mx.arange(logits.shape[-1])[None, :]
    forbid_text = (timestamp_logprob > max_text_logprob) & (columns < begin)
    return mx.where(forbid_text, -mx.inf, logits)


def apply() -> None:
    from mlx_whisper import decoding

    decoding.ApplyTimestampRules.apply = apply_timestamp_rules
