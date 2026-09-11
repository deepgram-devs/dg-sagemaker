"""Shared helpers for the Flux TTS (`/v2/speak`) e2e drivers.

Deliberately standalone rather than importing from `python-tts/e2e` — each
family's `e2e/` tree is its own package in this repo (see the note in
`python-tts/e2e/e2e_test_common.py`), so the audio-analysis and reporting
helpers are duplicated. Keep them in sync if the semantics change.
"""

from __future__ import annotations

import io
import math
import wave
from array import array

# ---------------------------------------------------------------------------
# Audio format the drivers request
# ---------------------------------------------------------------------------
#
# linear16 @ 24 kHz because that is the model's native output rate. Asking for
# anything else makes the server transcode, which adds a variable that has
# nothing to do with whether the endpoint works.
SAMPLE_RATE = 24000
ENCODING = "linear16"

# A response whose RMS is under this is silence, not speech. Same floor the
# Aura-2 driver uses. The point of the check: a voice that fails to load can
# still return the right NUMBER of bytes, just of nothing — which otherwise
# looks like a pass.
SILENCE_RMS_FLOOR = 150.0

# Supported `speed` range for Flux TTS: 0.5-1.5 in 0.05 increments, WIDENED from
# 0.85-1.15 by the 2026-08-31 release
# (https://developers.deepgram.com/changelog/2026/8/31). Every previously
# accepted value still works, so this is a pure widening. Applies to the WS
# streaming endpoint, the batch REST endpoint and Voice Agent `provider.version`
# v2 alike. Aura is a DIFFERENT range (0.7-1.5) and was not changed — do not
# reuse these constants for aura-2.
#
# Two distinct rejections, worth keeping apart when reading a failure:
#   SPEED_OUT_OF_RANGE      outside 0.5-1.5
#   SPEED_INCREMENT_INVALID inside the range but not a multiple of 0.05
#
# Keeping this stale cost a real false positive: the pre-widening test probed
# 1.3 as "out of range", and when the endpoint (correctly) accepted it the
# suite reported a product bug that did not exist.
MIN_SPEED = 0.5
MAX_SPEED = 1.5
SPEED_INCREMENT = 0.05
# Outside the range AND on-increment, so it provokes SPEED_OUT_OF_RANGE rather
# than SPEED_INCREMENT_INVALID — the negative control has to test one thing.
OUT_OF_RANGE_SPEED = 1.75
# Inside the range but off-increment, for the increment check.
OFF_INCREMENT_SPEED = 1.07

# The speed scenarios take this many PAIRED renders (each pair is its own
# default + fast clip, so ratios are internally paired and drift between
# iterations cancels) and judge the MEDIAN. Synthesis is stochastic — the model
# redistributes pauses differently each run — so one pair is not a verdict: a
# single sample read 1.03 on 2026-09-10 and was briefly taken for a broken
# speed control.
#
# Odd count on purpose, so the median is a real sample rather than an average
# of the middle two. Three tolerates ONE pathological render, which is the
# observed failure mode; raise to 5 if two-outlier runs ever show up, at the
# cost of 2 more synthesis round-trips per scenario.
SPEED_SAMPLES = 3
# Median-ratio gate. The ideal at MAX_SPEED=1.5 is 1/1.5 = 0.67; 0.85 leaves a
# wide buffer for prosodic variance while still sitting far below 1.0, which is
# what "the knob did nothing" looks like. Tightening toward 0.67 would start
# testing the model's pause distribution rather than whether speed works.
SPEED_RATIO_GATE = 0.85

# ---------------------------------------------------------------------------
# Test text
# ---------------------------------------------------------------------------

REFERENCE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Deepgram's Flux text to speech model streams audio one turn at a time, "
    "so a voice agent can start speaking before the whole sentence exists."
)

# Split so each element is a plausible LLM token burst — the turn-based API is
# designed to be fed incrementally, and feeding it one big string does not
# exercise the interesting path.
REFERENCE_PHRASES = [
    "Hi there! ",
    "I can help you with that. ",
    "Let me pull up your account details ",
    "and then we'll get it sorted out.",
]

# Long-ish single turn — checks that a turn spanning many segments still ends
# with exactly one SpeechMetadata.
LONG_TEXT = " ".join([REFERENCE_TEXT] * 4)


# ---------------------------------------------------------------------------
# Audio analysis
# ---------------------------------------------------------------------------

def sniff_container(data: bytes) -> str:
    """Best-effort container detection from magic bytes: wav / ogg / flac /
    mpeg / raw."""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"OggS":
        return "ogg"
    if data[:4] == b"fLaC":
        return "flac"
    if data[:3] == b"ID3":
        return "mpeg"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return "mpeg"
    return "raw"


def parse_wav(data: bytes) -> dict:
    """Parse a WAV container → {sample_rate, channels, sampwidth, pcm}."""
    with wave.open(io.BytesIO(data), "rb") as wf:
        return {
            "sample_rate": wf.getframerate(),
            "channels": wf.getnchannels(),
            "sampwidth": wf.getsampwidth(),
            "pcm": wf.readframes(wf.getnframes()),
        }


def analyze_pcm16(pcm: bytes, sample_rate: int) -> dict:
    """{n_samples, duration_s, rms, peak} for signed 16-bit mono PCM."""
    usable = len(pcm) - (len(pcm) % 2)
    samples = array("h")
    samples.frombytes(pcm[:usable])
    n = len(samples)
    if n == 0:
        return {"n_samples": 0, "duration_s": 0.0, "rms": 0.0, "peak": 0}
    return {
        "n_samples": n,
        "duration_s": n / sample_rate,
        "rms": math.sqrt(sum(s * s for s in samples) / n),
        "peak": max(abs(s) for s in samples),
    }


def linear16_duration_and_rms(data: bytes, requested_sample_rate: int) -> dict:
    """Duration + RMS for a linear16 response, WAV-wrapped or bare PCM."""
    container = sniff_container(data)
    if container == "wav":
        w = parse_wav(data)
        pcm, sr = w["pcm"], w["sample_rate"]
    else:
        pcm, sr = data, requested_sample_rate
    stats = analyze_pcm16(pcm, sr)
    return {
        "container": container,
        "sample_rate": sr,
        "duration_s": stats["duration_s"],
        "rms": stats["rms"],
        "peak": stats["peak"],
    }


def audio_verdict(data: bytes, sample_rate: int = SAMPLE_RATE) -> tuple[bool, str, dict]:
    """`(ok, note, stats)` for a synthesis response.

    Fails on empty audio AND on audio that is present but silent — the latter is
    the failure mode a byte-count check misses.
    """
    if not data:
        return False, "no audio bytes", {}
    stats = linear16_duration_and_rms(data, sample_rate)
    if stats["duration_s"] <= 0:
        return False, "zero-duration audio", stats
    if stats["rms"] < SILENCE_RMS_FLOOR:
        return (
            False,
            f"silent audio (rms {stats['rms']:.0f} < {SILENCE_RMS_FLOOR:.0f})",
            stats,
        )
    return True, "", stats


# ---------------------------------------------------------------------------
# Per-turn accounting cross-check
# ---------------------------------------------------------------------------

def check_session_totals(turns: list[dict], session: dict | None) -> str:
    """Cross-check `SessionMetadata` against the per-turn `SpeechMetadata`.

    Returns "" when consistent, else a short note. This is the client-side half
    of the billing check: metering counts the same characters that feed
    `SessionMetadata.total_billable_character_count`, so if the per-turn numbers
    don't sum to the session total, the metered number is suspect too.
    """
    if session is None:
        return "no SessionMetadata"
    if not turns:
        return ""
    want_billable = sum(t.get("billable_character_count", 0) for t in turns)
    want_input = sum(t.get("input_character_count", 0) for t in turns)
    got_billable = session.get("total_billable_character_count", 0)
    got_input = session.get("total_input_character_count", 0)
    problems = []
    if want_billable != got_billable:
        problems.append(f"billable {want_billable}!={got_billable}")
    if want_input != got_input:
        problems.append(f"input {want_input}!={got_input}")
    return "; ".join(problems)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary_table(rows: list[dict]) -> tuple[int, int]:
    """Render the per-scenario summary. Returns `(passed, failed)`.

    A row with `skipped=True` prints SKIP and counts toward neither.
    """
    if not rows:
        print("(no scenarios ran)")
        return 0, 0

    cols = [
        ("scenario", 30, "scenario"),
        ("status", 7, "ok"),
        ("bytes", 10, "bytes"),
        ("dur", 8, "duration_s"),
        ("rms", 9, "rms"),
        ("chars", 7, "billable_chars"),
        ("elapsed", 10, "elapsed_s"),
        ("notes", 44, "notes"),
    ]
    header = "  ".join(f"{title:<{w}}" for title, w, _ in cols)
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    passed = failed = skipped = 0
    for r in rows:
        is_skip = bool(r.get("skipped"))
        ok = bool(r.get("ok"))
        if is_skip:
            skipped += 1
        else:
            passed += ok
            failed += not ok
        cells = []
        for _, w, key in cols:
            v = r.get(key, "")
            if key == "ok":
                v = "SKIP" if is_skip else ("PASS" if ok else "FAIL")
            elif key == "elapsed_s" and isinstance(v, (int, float)):
                v = f"{v:.2f}s"
            elif key == "duration_s" and isinstance(v, (int, float)):
                v = f"{v:.2f}s" if v else ""
            elif key == "rms" and isinstance(v, (int, float)):
                v = f"{v:.0f}" if v else ""
            cells.append(f"{str(v):<{w}}"[:w])
        print("  ".join(cells))

    print("-" * len(header))
    tail = f" skipped={skipped}" if skipped else ""
    print(f"passed={passed} failed={failed}{tail}")
    print("=" * len(header))
    return passed, failed


def select_scenarios(all_names: list[str], requested: str | None) -> list[str]:
    """Resolve a `--scenarios a,b,c` selection against the battery.

    An unknown name is a hard error rather than a silent skip — a typo'd
    scenario list that quietly runs nothing reads as a clean pass.
    """
    if not requested:
        return all_names
    want = [s.strip() for s in requested.split(",") if s.strip()]
    unknown = [w for w in want if w not in all_names]
    if unknown:
        raise SystemExit(
            f"unknown scenario(s) {unknown}. Available: {all_names}"
        )
    return want
