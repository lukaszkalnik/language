"""Real backends; not run by default:

    uv run --package translate pytest -m integration tools/translate
"""

import pytest

from translate.core import translate_document
from translate.document import parse

pytestmark = pytest.mark.integration


def test_nllb_french_to_german():
    from translate.nllb import NLLBTranslator

    doc = parse("Bonjour tout le monde.\nMerci beaucoup.")
    translate_document(doc, NLLBTranslator("600m"), "de", source="fr", progress=False)
    out = " ".join(u.translation for u in doc.content()).lower()
    assert "danke" in out


def test_ollama_line_alignment():
    from translate.llm import OllamaTranslator

    doc = parse("Alors on danse\nQui dit étude dit travail\nQui dit taf te dit les thunes")
    translate_document(doc, OllamaTranslator(), "de", source="fr", progress=False)
    assert all(u.translation for u in doc.content())


def test_cached_nllb_needs_no_network():
    from ingest.testing import run_twice_second_trapped

    run, requests = run_twice_second_trapped("translate.cli", "--translator", "nllb", "--from", "fr", "--to", "de",
                                             stdin="Merci beaucoup.\n")
    assert run.returncode == 0, run.stderr
    assert "danke" in run.stdout.lower()
    assert requests == []
