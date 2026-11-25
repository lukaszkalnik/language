"""Hugging Face Whisper checkpoints (transformers layout) -> the layouts mlx-whisper and faster-whisper load.

Fine-tuned Whisper models such as KBLab's KB-Whisper or the Breton Whisper are published in the
transformers layout, not for MLX (and some not for CTranslate2 either). For MLX the weights are
the same tensors under other names, so converting them takes seconds once the checkpoint is
downloaded; CTranslate2 has its own converter.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

HF_TO_MLX = (  # applied in order
    ("model.", ""),
    (".layers.", ".blocks."),
    (".self_attn_layer_norm", ".attn_ln"),
    (".self_attn.", ".attn."),
    (".encoder_attn_layer_norm", ".cross_attn_ln"),
    (".encoder_attn.", ".cross_attn."),
    (".final_layer_norm", ".mlp_ln"),
    (".q_proj", ".query"),
    (".k_proj", ".key"),
    (".v_proj", ".value"),
    (".out_proj", ".out"),
    (".fc1", ".mlp1"),
    (".fc2", ".mlp2"),
    ("decoder.embed_positions.weight", "decoder.positional_embedding"),
    ("decoder.embed_tokens", "decoder.token_embedding"),
    ("encoder.layer_norm", "encoder.ln_post"),
    ("decoder.layer_norm", "decoder.ln"),
)
# Not parameters in MLX: the output projection is the token embedding, and the encoder's
# positions are fixed sinusoids computed at load time.
DROPPED = ("proj_out.weight", "encoder.embed_positions.weight")


def mlx_config(hf: dict) -> dict:
    return {
        "n_mels": hf["num_mel_bins"],
        "n_audio_ctx": hf["max_source_positions"],
        "n_audio_state": hf["d_model"],
        "n_audio_head": hf["encoder_attention_heads"],
        "n_audio_layer": hf["encoder_layers"],
        "n_vocab": hf["vocab_size"],
        "n_text_ctx": hf["max_target_positions"],
        "n_text_state": hf["d_model"],
        "n_text_head": hf["decoder_attention_heads"],
        "n_text_layer": hf["decoder_layers"],
        "model_type": "whisper",
    }


def mlx_name(hf_name: str) -> str | None:
    """The MLX parameter name of a transformers Whisper tensor (None: not needed)."""
    name = hf_name
    for old, new in HF_TO_MLX:
        name = name.replace(old, new)
    return None if name in DROPPED else name


def convert_weights(weights: dict, dtype=None) -> dict:
    """Rename transformers tensors to MLX names; convolutions go from (out, in, k) to (out, k, in)."""
    out = {}
    for key, value in weights.items():
        name = mlx_name(key)
        if name is None:
            continue
        if ".conv" in name and name.endswith(".weight") and value.ndim == 3:
            value = value.swapaxes(1, 2)
        out[name] = value.astype(dtype) if dtype is not None else value
    return out


def convert_checkpoint(src: Path, dst: Path) -> Path:
    """Write config.json and weights.safetensors for mlx-whisper from a transformers checkpoint."""
    import mlx.core as mx

    hf_config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    weights = convert_weights(mx.load(str(src / "model.safetensors")), mx.float16)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    mx.save_safetensors(str(tmp / "weights.safetensors"), weights)
    (tmp / "config.json").write_text(json.dumps(mlx_config(hf_config), indent=2), encoding="utf-8")
    shutil.rmtree(dst, ignore_errors=True)
    tmp.replace(dst)
    return dst


def convert_ct2(src: Path, dst: Path) -> Path:
    """A CTranslate2 copy for faster-whisper (other platforms than Apple Silicon), in float16.

    Needs transformers and torch; faster-whisper converts the weights to int8 or float16 as it
    loads them, depending on the device.
    """
    from ctranslate2.converters import TransformersConverter

    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    TransformersConverter(str(src), copy_files=["tokenizer.json"]).convert(str(tmp), quantization="float16", force=True)
    shutil.rmtree(dst, ignore_errors=True)
    tmp.replace(dst)
    return dst
