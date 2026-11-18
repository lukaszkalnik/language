import numpy as np
import pytest

from accent.classify import DISPLAY_NAMES, average_probabilities, display_name, softmax, top_k
from accent.cli import NOTE, render_text


def test_softmax_sums_to_one():
    p = softmax(np.array([0.1, 0.5, 0.2]) * 30)
    assert p.sum() == pytest.approx(1.0)
    assert p.argmax() == 1


def test_weighted_average():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    np.testing.assert_allclose(average_probabilities([a, b]), [0.5, 0.5])
    np.testing.assert_allclose(average_probabilities([a, b], [3, 1]), [0.75, 0.25])
    with pytest.raises(ValueError):
        average_probabilities([])


def test_top_k_order():
    labels = ["us", "england", "indian"]
    assert top_k(np.array([0.2, 0.7, 0.1]), labels, 2) == [("england", 0.7), ("us", 0.2)]


def test_all_sixteen_labels_have_names():
    assert len(DISPLAY_NAMES) == 16
    assert display_name("southatlandtic") == "South Atlantic"
    assert display_name("unknown") == "unknown"


def test_render_text_and_note():
    result = {
        "model": "ecapa",
        "summary": {"chunks": 3, "silent_chunks": 1, "analysed_seconds": 38.0},
        "top": [{"name": "Irish", "probability": 0.6}, {"name": "Scottish", "probability": 0.3},
                {"name": "Welsh", "probability": 0.1}],
        "chunks": [
            {"index": 1, "start": 0, "end": 20, "rms_dbfs": -20.0, "speech_seconds": 19, "silent": False,
             "top_name": "Irish", "top_probability": 0.7},
            {"index": 2, "start": 20, "end": 40, "rms_dbfs": None, "speech_seconds": 0, "silent": True},
        ],
    }
    text = render_text(result, verbose=True)
    assert "1. Irish" in text and "60.0%" in text
    assert "(silent, skipped)" in text
    assert "statistical guess" in NOTE and "misclassified" in NOTE
