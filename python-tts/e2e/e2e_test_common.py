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
"""

from __future__ import annotations

import io
import math
import wave
from array import array

# A known, length-stable sentence. ~95 chars / ~17 words — long enough that the
# synthesized clip is comfortably over a second at normal speed, so duration
# comparisons (speed control) and RMS checks are stable.
REFERENCE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Deepgram converts this sentence into natural sounding speech on Amazon SageMaker."
)

# Multi-phrase fixture for the streaming driver (one Speak+Flush per phrase).
REFERENCE_PHRASES = [
    "The quick brown fox jumps over the lazy dog.",
    "Deepgram text to speech runs on Amazon SageMaker.",
    "This is the third and final test phrase for the streaming run.",
]

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
