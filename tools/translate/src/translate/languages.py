"""Language codes: ISO 639-1 <-> English names <-> NLLB-200 codes, and plain-text detection."""

from __future__ import annotations

# code: (English name, NLLB-200 code)
LANGUAGES = {
    "af": ("Afrikaans", "afr_Latn"), "ar": ("Arabic", "arb_Arab"), "bg": ("Bulgarian", "bul_Cyrl"),
    "br": ("Breton", None),  # not in NLLB-200: translated into French first (breton.py)
    "bn": ("Bengali", "ben_Beng"), "ca": ("Catalan", "cat_Latn"), "cs": ("Czech", "ces_Latn"),
    "cy": ("Welsh", "cym_Latn"), "da": ("Danish", "dan_Latn"), "de": ("German", "deu_Latn"),
    "el": ("Greek", "ell_Grek"), "en": ("English", "eng_Latn"), "es": ("Spanish", "spa_Latn"),
    "et": ("Estonian", "est_Latn"), "eu": ("Basque", "eus_Latn"), "fa": ("Persian", "pes_Arab"),
    "fi": ("Finnish", "fin_Latn"), "fr": ("French", "fra_Latn"), "ga": ("Irish", "gle_Latn"),
    "gl": ("Galician", "glg_Latn"), "he": ("Hebrew", "heb_Hebr"), "hi": ("Hindi", "hin_Deva"),
    "hr": ("Croatian", "hrv_Latn"), "hu": ("Hungarian", "hun_Latn"), "hy": ("Armenian", "hye_Armn"),
    "id": ("Indonesian", "ind_Latn"), "is": ("Icelandic", "isl_Latn"), "it": ("Italian", "ita_Latn"),
    "ja": ("Japanese", "jpn_Jpan"), "ka": ("Georgian", "kat_Geor"), "kk": ("Kazakh", "kaz_Cyrl"),
    "ko": ("Korean", "kor_Hang"), "lt": ("Lithuanian", "lit_Latn"), "lv": ("Latvian", "lvs_Latn"),
    "mk": ("Macedonian", "mkd_Cyrl"), "ms": ("Malay", "zsm_Latn"), "mt": ("Maltese", "mlt_Latn"),
    "nl": ("Dutch", "nld_Latn"), "no": ("Norwegian", "nob_Latn"), "nb": ("Norwegian Bokmål", "nob_Latn"),
    "pl": ("Polish", "pol_Latn"), "pt": ("Portuguese", "por_Latn"), "ro": ("Romanian", "ron_Latn"),
    "ru": ("Russian", "rus_Cyrl"), "sk": ("Slovak", "slk_Latn"), "sl": ("Slovenian", "slv_Latn"),
    "sq": ("Albanian", "als_Latn"), "sr": ("Serbian", "srp_Cyrl"), "sv": ("Swedish", "swe_Latn"),
    "sw": ("Swahili", "swh_Latn"), "ta": ("Tamil", "tam_Taml"), "th": ("Thai", "tha_Thai"),
    "tl": ("Tagalog", "tgl_Latn"), "tr": ("Turkish", "tur_Latn"), "uk": ("Ukrainian", "ukr_Cyrl"),
    "ur": ("Urdu", "urd_Arab"), "vi": ("Vietnamese", "vie_Latn"), "yo": ("Yoruba", "yor_Latn"),
    "zh": ("Chinese", "zho_Hans"), "yue": ("Cantonese", "yue_Hant"),
}


def normalise(code: str) -> str:
    c = code.strip().lower().replace("_", "-")
    if c in LANGUAGES:
        return c
    base = c.split("-")[0]
    if base in LANGUAGES:
        return base
    for k, (name, nllb) in LANGUAGES.items():
        if c in (name.lower(), (nllb or "").lower()):
            return k
    raise ValueError(f"unknown language {code!r}; use a code such as de, en, pl, fr")


def name(code: str | None) -> str:
    if not code:
        return "the source language"
    return LANGUAGES.get(code, (code, None))[0]


def nllb_code(code: str) -> str:
    if not LANGUAGES.get(code, (None, None))[1]:
        raise ValueError(f"NLLB has no mapping for language {code!r}")
    return LANGUAGES[code][1]


def detect(text: str) -> str | None:
    """Detect the language of plain text (py3langid, offline). None if there is too little text."""
    if len(text.strip()) < 12:
        return None
    import py3langid

    code, _ = py3langid.classify(text)
    return code if code in LANGUAGES else None
