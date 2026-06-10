"""Shared helpers for the e2e SageMaker TTS correctness tests.

Unlike STT (where Word Error Rate against a known transcript is the pass/fail
signal), text-to-speech has no transcript to score against, so these e2e tests
validate the **synthesized audio itself**, self-contained — no second endpoint
required:

- the response is non-empty,
- its container/codec matches what was requested (WAV / Ogg / FLAC / MPEG
  magic bytes; raw PCM for `container=none`),
- for `linear16` PCM: the audio is non-silent (RMS above a floor) and its
  sample rate + duration are plausible for the input text,
- the requested `speed` measurably changes duration for the same text.

This module is pure-Python and has no AWS dependency — it is import-safe from
the batch driver, the streaming driver, and unit tests alike.

Language + voice awareness
--------------------------
TTS bundles are monolingual (one Aura-2 text2codes pack per language; e.g.
``aura-2-sirio-es`` won't load on an EN-only endpoint and vice versa). To
exercise a SageMaker TTS endpoint correctly the harness needs (a) reference
text in the **right language** and (b) a slate of **language-matched voices**
to demonstrate that multiple voices load and synthesize from the bundle. Both
live in this module so the streaming and batch drivers share one source of
truth, kept in lockstep with the public voice catalog at
https://developers.deepgram.com/docs/tts-models.
"""

from __future__ import annotations

import io
import math
import wave
from array import array
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Per-language reference text fixtures
# ---------------------------------------------------------------------------
#
# Each language entry holds a single-sentence reference paragraph (~90–150 chars
# / ~15–25 words) and a 3-phrase list for the streaming `multi_phrase_flush`
# scenario. The single-paragraph form runs comfortably over a second at normal
# speed, so RMS + duration checks are stable. Content is in the target language
# so the voice's phonemes match the script — the previous English-only fixtures
# silently passed on non-EN endpoints by reading English-as-target-language,
# which is *not* a real language-coverage test.
#
# Add a new language by appending to both maps + ``VOICE_CATALOG``. Voices
# follow ``aura-2-<name>-<lang>`` and the harness derives language from the
# trailing suffix (see ``voice_language``).

LANGUAGE_TEXTS: dict[str, str] = {
    "en": (
        "The quick brown fox jumps over the lazy dog. "
        "Deepgram converts this sentence into natural sounding speech on Amazon SageMaker."
    ),
    "es": (
        "El veloz murciélago hindú comía feliz cardillo y kiwi. "
        "Deepgram convierte esta frase en habla natural en Amazon SageMaker."
    ),
    "de": (
        "Der schnelle braune Fuchs springt über den faulen Hund. "
        "Deepgram wandelt diesen Satz in natürliche Sprache auf Amazon SageMaker um."
    ),
    "fr": (
        "Le rapide renard brun saute par-dessus le chien paresseux. "
        "Deepgram convertit cette phrase en parole naturelle sur Amazon SageMaker."
    ),
    "it": (
        "La veloce volpe marrone salta sopra il cane pigro. "
        "Deepgram converte questa frase in voce naturale su Amazon SageMaker."
    ),
    "nl": (
        "De snelle bruine vos springt over de luie hond. "
        "Deepgram zet deze zin om in natuurlijke spraak op Amazon SageMaker."
    ),
    "ja": (
        "素早い茶色のキツネが怠惰な犬を飛び越えます。"
        "Deepgram は、この文を Amazon SageMaker で自然な音声に変換します。"
    ),
}

LANGUAGE_PHRASES: dict[str, list[str]] = {
    "en": [
        "The quick brown fox jumps over the lazy dog.",
        "Deepgram text to speech runs on Amazon SageMaker.",
        "This is the third and final test phrase for the streaming run.",
    ],
    "es": [
        "El veloz murciélago hindú comía feliz cardillo y kiwi.",
        "El servicio de texto a voz de Deepgram funciona en Amazon SageMaker.",
        "Esta es la tercera y última frase de prueba para la transmisión.",
    ],
    "de": [
        "Der schnelle braune Fuchs springt über den faulen Hund.",
        "Deepgrams Text-zu-Sprache läuft auf Amazon SageMaker.",
        "Dies ist der dritte und letzte Testsatz für den Streaming-Lauf.",
    ],
    "fr": [
        "Le rapide renard brun saute par-dessus le chien paresseux.",
        "Le service texte vers parole de Deepgram fonctionne sur Amazon SageMaker.",
        "Ceci est la troisième et dernière phrase de test pour la diffusion.",
    ],
    "it": [
        "La veloce volpe marrone salta sopra il cane pigro.",
        "Il servizio da testo a voce di Deepgram funziona su Amazon SageMaker.",
        "Questa è la terza e ultima frase di prova per lo streaming.",
    ],
    "nl": [
        "De snelle bruine vos springt over de luie hond.",
        "De tekst-naar-spraak van Deepgram draait op Amazon SageMaker.",
        "Dit is de derde en laatste testzin voor de streaming-run.",
    ],
    "ja": [
        "素早い茶色のキツネが怠惰な犬を飛び越えます。",
        "Deepgram のテキスト読み上げは Amazon SageMaker で動作します。",
        "これはストリーミング実行の3番目で最後のテスト文です。",
    ],
}

# Back-compat aliases for older callers (the English defaults).
REFERENCE_TEXT = LANGUAGE_TEXTS["en"]
REFERENCE_PHRASES = LANGUAGE_PHRASES["en"]

# An inline IPA pronunciation override (Aura-2 voice control). The escaped-brace
# object is embedded directly in the text; the synthesizer should accept it and
# produce audio (auditory correctness is not asserted — this is a smoke check).
# See https://developers.deepgram.com/docs/tts-voice-controls
IPA_TEXT = (
    'The medication \\{"word":"dupilumab","pronounce":"duːˈpɪljuːmæb"\\} '
    "is administered every two weeks."
)

# int16 RMS below this is effectively silence — a healthy TTS clip is far above.
SILENCE_RMS_FLOOR = 150.0


# ---------------------------------------------------------------------------
# Voice catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Voice:
    """One entry in the public Aura voice catalog."""
    model: str            # e.g. "aura-2-sirio-es"
    language: str         # ISO code, e.g. "es"
    accent: str           # natural-language label, e.g. "Mexican"
    gender: str           # "F" | "M"
    featured: bool = False  # marked "Featured" in the public docs


# Voices ordered FEATURED FIRST then by docs order. Featured voices are what the
# harness uses for the multi-voice coverage scenario (`featured_voices`). When a
# new voice is added to https://developers.deepgram.com/docs/tts-models, append
# here and re-run the e2e against a bundle that ships it.
VOICE_CATALOG: dict[str, list[Voice]] = {
    "en": [
        # Aura-2 — 6 voices flagged "Featured" in the docs.
        Voice("aura-2-thalia-en",     "en", "American", "F", featured=True),
        Voice("aura-2-andromeda-en",  "en", "American", "F", featured=True),
        Voice("aura-2-helena-en",     "en", "American", "F", featured=True),
        Voice("aura-2-apollo-en",     "en", "American", "M", featured=True),
        Voice("aura-2-arcas-en",      "en", "American", "M", featured=True),
        Voice("aura-2-aries-en",      "en", "American", "M", featured=True),
        # Rest of Aura-2 (sample — extend as needed; harness only consumes the
        # featured slice unless --voice-coverage-n exceeds it).
        Voice("aura-2-amalthea-en",   "en", "Filipino", "F"),
        Voice("aura-2-asteria-en",    "en", "American", "F"),
        Voice("aura-2-orion-en",      "en", "American", "M"),
        Voice("aura-2-zeus-en",       "en", "American", "M"),
        Voice("aura-2-draco-en",      "en", "British",  "M"),
        Voice("aura-2-pandora-en",    "en", "British",  "F"),
        Voice("aura-2-hyperion-en",   "en", "Australian", "M"),
        Voice("aura-2-theia-en",      "en", "Australian", "F"),
    ],
    "es": [
        # 3 Featured per docs.
        Voice("aura-2-celeste-es",    "es", "Colombian",     "F", featured=True),
        Voice("aura-2-estrella-es",   "es", "Mexican",       "F", featured=True),
        Voice("aura-2-nestor-es",     "es", "Peninsular",    "M", featured=True),
        # Remaining 14 from the Spanish Aura-2 lineup.
        Voice("aura-2-sirio-es",      "es", "Mexican",       "M"),
        Voice("aura-2-carina-es",     "es", "Peninsular",    "F"),
        Voice("aura-2-alvaro-es",     "es", "Peninsular",    "M"),
        Voice("aura-2-diana-es",      "es", "Peninsular",    "F"),
        Voice("aura-2-aquila-es",     "es", "Latin American", "M"),
        Voice("aura-2-selena-es",     "es", "Latin American", "F"),
        Voice("aura-2-javier-es",     "es", "Mexican",       "M"),
        Voice("aura-2-agustina-es",   "es", "Peninsular",    "F"),
        Voice("aura-2-antonia-es",    "es", "Argentine",     "F"),
        Voice("aura-2-gloria-es",     "es", "Colombian",     "F"),
        Voice("aura-2-luciano-es",    "es", "Mexican",       "M"),
        Voice("aura-2-olivia-es",     "es", "Mexican",       "F"),
        Voice("aura-2-silvia-es",     "es", "Peninsular",    "F"),
        Voice("aura-2-valerio-es",    "es", "Mexican",       "M"),
    ],
    "de": [
        # Docs do not mark a "Featured" subset; harness treats the first 3 as
        # canonical.
        Voice("aura-2-julius-de",     "de", "German", "M"),
        Voice("aura-2-viktoria-de",   "de", "German", "F"),
        Voice("aura-2-elara-de",      "de", "German", "F"),
        Voice("aura-2-aurelia-de",    "de", "German", "F"),
        Voice("aura-2-lara-de",       "de", "German", "F"),
        Voice("aura-2-fabian-de",     "de", "German", "M"),
        Voice("aura-2-kara-de",       "de", "German", "F"),
    ],
    "fr": [
        # Only 2 French voices documented.
        Voice("aura-2-agathe-fr",     "fr", "French", "F"),
        Voice("aura-2-hector-fr",     "fr", "French", "M"),
    ],
    "it": [
        Voice("aura-2-livia-it",      "it", "Italian", "F"),
        Voice("aura-2-dionisio-it",   "it", "Italian", "M"),
        Voice("aura-2-melia-it",      "it", "Italian", "F"),
        Voice("aura-2-elio-it",       "it", "Italian", "M"),
        Voice("aura-2-flavio-it",     "it", "Italian", "M"),
        Voice("aura-2-maia-it",       "it", "Italian", "F"),
        Voice("aura-2-cinzia-it",     "it", "Italian", "F"),
        Voice("aura-2-cesare-it",     "it", "Italian", "M"),
        Voice("aura-2-perseo-it",     "it", "Italian", "M"),
        Voice("aura-2-demetra-it",    "it", "Italian", "F"),
    ],
    "nl": [
        Voice("aura-2-rhea-nl",       "nl", "Dutch", "F"),
        Voice("aura-2-sander-nl",     "nl", "Dutch", "M"),
        Voice("aura-2-beatrix-nl",    "nl", "Dutch", "F"),
        Voice("aura-2-daphne-nl",     "nl", "Dutch", "F"),
        Voice("aura-2-cornelia-nl",   "nl", "Dutch", "F"),
        Voice("aura-2-hestia-nl",     "nl", "Dutch", "F"),
        Voice("aura-2-lars-nl",       "nl", "Dutch", "M"),
        Voice("aura-2-roman-nl",      "nl", "Dutch", "M"),
        Voice("aura-2-leda-nl",       "nl", "Dutch", "F"),
    ],
    "ja": [
        Voice("aura-2-fujin-ja",      "ja", "Japanese", "M"),
        Voice("aura-2-izanami-ja",    "ja", "Japanese", "F"),
        Voice("aura-2-uzume-ja",      "ja", "Japanese", "F"),
        Voice("aura-2-ebisu-ja",      "ja", "Japanese", "M"),
        Voice("aura-2-ama-ja",        "ja", "Japanese", "F"),
    ],
}

SUPPORTED_LANGUAGES: tuple[str, ...] = tuple(VOICE_CATALOG.keys())


# ---------------------------------------------------------------------------
# Language / voice helpers
# ---------------------------------------------------------------------------

def voice_language(voice: str) -> str | None:
    """Derive the ISO language code from a voice model string.

    ``aura-2-sirio-es`` → ``"es"``. Returns None if the suffix isn't a known
    language (which is the safe behavior — callers should fall back to the
    explicit ``--language`` argument).
    """
    if not voice or "-" not in voice:
        return None
    suffix = voice.rsplit("-", 1)[-1].lower()
    return suffix if suffix in VOICE_CATALOG else None


def reference_text(language: str) -> str:
    """Reference paragraph for the given language."""
    if language not in LANGUAGE_TEXTS:
        raise KeyError(f"no reference text for language {language!r}; "
                       f"known: {sorted(LANGUAGE_TEXTS)}")
    return LANGUAGE_TEXTS[language]


def reference_phrases(language: str) -> list[str]:
    """3-phrase fixture for the streaming multi-phrase scenario."""
    if language not in LANGUAGE_PHRASES:
        raise KeyError(f"no reference phrases for language {language!r}; "
                       f"known: {sorted(LANGUAGE_PHRASES)}")
    return list(LANGUAGE_PHRASES[language])


def default_voice(language: str) -> str:
    """The first featured (or, lacking a featured marker, first listed) voice
    for the language — used as the default when ``--voice`` is omitted."""
    voices = VOICE_CATALOG.get(language)
    if not voices:
        raise KeyError(f"no voices for language {language!r}; "
                       f"known: {sorted(VOICE_CATALOG)}")
    featured = [v for v in voices if v.featured]
    return (featured or voices)[0].model


def featured_voices(language: str, n: int = 3) -> list[Voice]:
    """The N voices to drive the multi-voice coverage scenario.

    Featured voices (docs-labelled) come first; if fewer than N are flagged,
    fall through to the rest of the catalog in docs order. Stable across runs
    so the per-voice coverage rows in the summary table are diff-friendly.
    """
    if language not in VOICE_CATALOG:
        raise KeyError(f"no voices for language {language!r}; "
                       f"known: {sorted(VOICE_CATALOG)}")
    voices = VOICE_CATALOG[language]
    featured = [v for v in voices if v.featured]
    rest = [v for v in voices if not v.featured]
    return (featured + rest)[:max(1, n)]


def alt_language_voice(language: str) -> Voice | None:
    """Pick a voice in a DIFFERENT language than ``language`` (negative test —
    expected to error on a monolingual bundle). Returns None if no alternative
    catalog exists (single-language deployment of the harness)."""
    for lang, voices in VOICE_CATALOG.items():
        if lang != language and voices:
            return voices[0]
    return None


# ---------------------------------------------------------------------------
# Container / codec sniffing
# ---------------------------------------------------------------------------

def sniff_container(data: bytes) -> str:
    """Best-effort container/codec detection from magic bytes.

    Returns one of: 'wav', 'ogg', 'flac', 'mpeg' (mp3 or AAC/ADTS), 'raw'.
    """
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"OggS":
        return "ogg"
    if data[:4] == b"fLaC":
        return "flac"
    if data[:3] == b"ID3":
        return "mpeg"
    # MPEG audio / AAC-ADTS frame sync: 11 set bits (0xFFE.. / 0xFFF..).
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return "mpeg"
    return "raw"


def parse_wav(data: bytes) -> dict:
    """Parse a WAV container. Returns {sample_rate, channels, sampwidth, pcm}.

    Raises wave.Error / EOFError if `data` is not a valid WAV.
    """
    with wave.open(io.BytesIO(data), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        pcm = wf.readframes(wf.getnframes())
    return {"sample_rate": sr, "channels": ch, "sampwidth": sw, "pcm": pcm}


def analyze_pcm16(pcm: bytes, sample_rate: int) -> dict:
    """Compute {n_samples, duration_s, rms, peak} for signed 16-bit mono PCM."""
    usable = len(pcm) - (len(pcm) % 2)
    samples = array("h")
    samples.frombytes(pcm[:usable])
    n = len(samples)
    if n == 0:
        return {"n_samples": 0, "duration_s": 0.0, "rms": 0.0, "peak": 0}
    peak = max(abs(s) for s in samples)
    rms = math.sqrt(sum(s * s for s in samples) / n)
    return {
        "n_samples": n,
        "duration_s": n / sample_rate,
        "rms": rms,
        "peak": peak,
    }


def linear16_duration_and_rms(data: bytes, requested_sample_rate: int) -> dict:
    """Duration + RMS for a linear16 response, whether WAV-wrapped or raw PCM.

    Returns {container, sample_rate, duration_s, rms, peak}.
    """
    container = sniff_container(data)
    if container == "wav":
        w = parse_wav(data)
        pcm = w["pcm"]
        sr = w["sample_rate"]
    else:
        pcm = data            # container=none → bare PCM frames
        sr = requested_sample_rate
    stats = analyze_pcm16(pcm, sr)
    return {
        "container": container,
        "sample_rate": sr,
        "duration_s": stats["duration_s"],
        "rms": stats["rms"],
        "peak": stats["peak"],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary_table(rows: list[dict]) -> tuple[int, int]:
    """Render the per-scenario TTS summary table. Returns (pass_count, fail_count)."""
    if not rows:
        print("(no scenarios ran)")
        return 0, 0

    cols = [
        ("scenario",     30, "scenario"),
        ("status",        7, "ok"),
        ("bytes",        10, "bytes"),
        ("dur",           8, "duration_s"),
        ("rms",           9, "rms"),
        ("elapsed",      10, "elapsed_s"),
        ("notes",        46, "notes"),
    ]
    header = "  ".join(f"{title:<{w}}" for title, w, _ in cols)
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    passed = failed = 0
    for r in rows:
        ok = bool(r.get("ok"))
        passed += ok
        failed += (not ok)
        cells = []
        for _, w, key in cols:
            v = r.get(key, "")
            if key == "ok":
                v = "PASS" if ok else "FAIL"
            elif key == "elapsed_s" and isinstance(v, (int, float)):
                v = f"{v:.2f}s"
            elif key == "duration_s" and isinstance(v, (int, float)):
                v = f"{v:.2f}s" if v else ""
            elif key == "rms" and isinstance(v, (int, float)):
                v = f"{v:.0f}" if v else ""
            else:
                v = str(v) if v is not None else ""
            cells.append(f"{v:<{w}}")
        print("  ".join(cells))
    print("=" * len(header))
    print(f"PASSED: {passed}  FAILED: {failed}  TOTAL: {len(rows)}")
    return passed, failed
