import numpy as np

from transcribe import whisper
from transcribe.convert import convert_weights, mlx_config, mlx_name
from transcribe.whisper import model_for


def test_swedish_gets_kb_whisper_unless_a_model_is_chosen():
    assert model_for("auto", "sv") == "kb-whisper"
    assert model_for("auto", "en") == "large-v3"
    assert model_for("auto", "sv", per_passage_languages=True) == "large-v3"
    assert model_for("large-v3", "sv") == "large-v3"
    assert model_for("turbo", "de") == "turbo"


def test_breton_gets_the_breton_whisper():
    assert model_for("auto", "br") == "whisper-br"
    assert model_for("auto", "fr") == "large-v3"
    assert model_for("auto", "br", per_passage_languages=True) == "large-v3"


def test_converted_copies_keep_their_cache_folders(tmp_path, monkeypatch):
    # KB-Whisper's MLX copy from earlier versions must be found again, not converted anew
    monkeypatch.setattr(whisper, "cache_dir", lambda *parts: tmp_path)
    for name, backend, probe in (("kb-whisper", "mlx", "weights.safetensors"), ("whisper-br", "ct2", "model.bin")):
        folder = tmp_path / f"{whisper.FINE_TUNES[name].folder}-{backend}"
        folder.mkdir()
        (folder / probe).write_bytes(b"")
        assert whisper.converted(name, backend) == str(folder)
    assert whisper.FINE_TUNES["kb-whisper"].folder + "-mlx" == "kb-whisper-large-d5d5984-mlx"


def test_every_model_has_a_description_and_pinned_revision():
    assert set(whisper.DESCRIPTIONS) == set(whisper.MODELS)
    assert all(len(t.revision) == 40 for t in whisper.FINE_TUNES.values())
    assert set(whisper.LANGUAGE_MODELS.values()) <= set(whisper.FINE_TUNES)


def test_transformers_names_become_mlx_names():
    assert mlx_name("model.encoder.layers.3.self_attn.q_proj.weight") == "encoder.blocks.3.attn.query.weight"
    assert mlx_name("model.decoder.layers.0.encoder_attn.out_proj.bias") == "decoder.blocks.0.cross_attn.out.bias"
    assert mlx_name("model.decoder.layers.1.encoder_attn_layer_norm.weight") == "decoder.blocks.1.cross_attn_ln.weight"
    assert mlx_name("model.decoder.layers.1.self_attn_layer_norm.bias") == "decoder.blocks.1.attn_ln.bias"
    assert mlx_name("model.encoder.layers.2.final_layer_norm.weight") == "encoder.blocks.2.mlp_ln.weight"
    assert mlx_name("model.encoder.layers.2.fc2.weight") == "encoder.blocks.2.mlp2.weight"
    assert mlx_name("model.decoder.embed_tokens.weight") == "decoder.token_embedding.weight"
    assert mlx_name("model.decoder.embed_positions.weight") == "decoder.positional_embedding"
    assert mlx_name("model.encoder.layer_norm.weight") == "encoder.ln_post.weight"
    assert mlx_name("model.decoder.layer_norm.bias") == "decoder.ln.bias"
    assert mlx_name("model.encoder.embed_positions.weight") is None
    assert mlx_name("proj_out.weight") is None


def test_convolutions_are_transposed_for_mlx():
    out = convert_weights({"model.encoder.conv1.weight": np.zeros((8, 4, 3)),
                           "model.encoder.conv1.bias": np.zeros(8), "proj_out.weight": np.zeros((2, 2))})
    assert out["encoder.conv1.weight"].shape == (8, 3, 4)
    assert out["encoder.conv1.bias"].shape == (8,)
    assert "proj_out.weight" not in out


def test_mlx_config_from_transformers_config():
    hf = {"num_mel_bins": 128, "max_source_positions": 1500, "d_model": 1280, "encoder_attention_heads": 20,
          "encoder_layers": 32, "vocab_size": 51866, "max_target_positions": 448,
          "decoder_attention_heads": 20, "decoder_layers": 32}
    cfg = mlx_config(hf)
    assert cfg["n_mels"] == 128 and cfg["n_text_state"] == 1280 and cfg["n_vocab"] == 51866
