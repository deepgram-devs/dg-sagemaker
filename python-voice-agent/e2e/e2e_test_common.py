"""Shared constants and helpers for the Voice Agent e2e driver and example client.

Deliberately standalone rather than importing from another family's `e2e/` —
each family's tree is its own package in this repo. The audio-analysis and
reporting helpers mirror the ones in `python-flux-tts/e2e/e2e_test_common.py`;
keep them in sync if the semantics change.
"""

from __future__ import annotations

import io
import math
import wave
from array import array

# ---------------------------------------------------------------------------
# Audio formats
# ---------------------------------------------------------------------------
#
# Caller audio goes in as linear16 at 16 kHz (what a phone/VoIP leg resampled to
# wideband typically is); the agent's voice comes back as linear16 at 24 kHz,
# the speech model's native rate, so the server does not resample on the way out.
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000

# Audio is sent in 100 ms frames at real-time pace. Sending faster than real
# time is not "a faster test": end-of-turn detection works on the audio itself,
# so a burst changes what the agent hears.
CHUNK_MS = 100
CHUNK_BYTES = INPUT_SAMPLE_RATE * 2 * CHUNK_MS // 1000
SILENCE_CHUNK = b"\0" * CHUNK_BYTES

# Agent audio whose RMS is under this is silence, not speech. Same floor the
# TTS drivers use: a voice that fails to load can still produce the right
# NUMBER of bytes, of nothing.
SILENCE_RMS_FLOOR = 150.0

# A reply shorter than this is not a spoken sentence.
MIN_SPOKEN_S = 0.5

DEFAULT_VOICE = "flux-alexis-en"

# ---------------------------------------------------------------------------
# Test content
# ---------------------------------------------------------------------------

GREETING = "Hi, thanks for calling. How can I help?"
PROMPT = (
    "You are a friendly restaurant host on a phone call. "
    "Keep every reply to one or two short spoken sentences."
)

# Caller lines, synthesized with the endpoint's own text-to-speech so the run
# needs no audio fixtures. Each carries words the transcription check looks for.
CALLER_LINES = [
    "Hi, do you have a table for four tomorrow evening at seven?",
    "Great. Can we sit outside on the patio?",
    "Perfect, please put it under the name Taylor.",
]
CALLER_KEYWORDS = {"table", "four", "tomorrow", "evening", "seven"}

# Long agent line for the barge-in scenario: long enough that the caller can
# interrupt well before it ends.
LONG_AGENT_LINE = (
    "Of course. Let me tell you about tonight's specials. We have a slow roasted "
    "chicken with lemon and thyme, a wild mushroom risotto finished with parmesan, "
    "grilled salmon over a warm lentil salad, and for dessert a dark chocolate tart "
    "with sea salt and whipped cream. Every main course comes with fresh bread."
)
INTERRUPT_LINE = "Sorry to interrupt, I just need the address please."

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def pcm16_stats(pcm: bytes, sample_rate: int) -> dict:
    """{duration_s, rms} for signed 16-bit mono PCM."""
    usable = len(pcm) - (len(pcm) % 2)
    samples = array("h")
    samples.frombytes(pcm[:usable])
    n = len(samples)
    if n == 0:
        return {"duration_s": 0.0, "rms": 0.0}
    return {
        "duration_s": n / sample_rate,
        "rms": math.sqrt(sum(s * s for s in samples) / n),
    }


def spoken_verdict(pcm: bytes, sample_rate: int = OUTPUT_SAMPLE_RATE) -> tuple[bool, str, dict]:
    """`(ok, note, stats)`: present, long enough to be speech, and not silent."""
    stats = pcm16_stats(pcm, sample_rate)
    if not pcm:
        return False, "no audio", stats
    if stats["duration_s"] < MIN_SPOKEN_S:
        return False, f"only {stats['duration_s']:.2f}s of audio", stats
    if stats["rms"] < SILENCE_RMS_FLOOR:
        return False, f"silent audio (rms {stats['rms']:.0f})", stats
    return True, "", stats


def strip_wav(data: bytes) -> tuple[bytes, int | None]:
    """`(pcm, sample_rate)` for a WAV container, or `(data, None)` for bare PCM."""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        with wave.open(io.BytesIO(data), "rb") as wf:
            return wf.readframes(wf.getnframes()), wf.getframerate()
    return data, None


def write_wav(path: str, pcm: bytes, sample_rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def read_wav_16k_mono(path: str) -> bytes:
    """PCM from a 16 kHz mono 16-bit WAV. Anything else is refused with the
    command that converts it — resampling here would hide a format mistake."""
    with wave.open(path, "rb") as wf:
        rate, ch, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        if (rate, ch, width) != (INPUT_SAMPLE_RATE, 1, 2):
            raise SystemExit(
                f"{path}: {rate} Hz, {ch} channel(s), {8 * width}-bit; need 16000 Hz mono 16-bit. "
                f"Convert with: ffmpeg -i {path} -ar 16000 -ac 1 -c:a pcm_s16le out.wav"
            )
        return wf.readframes(wf.getnframes())


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary_table(rows: list[dict]) -> tuple[int, int]:
    """Render the per-scenario summary and the final PASSED block.

    Returns `(passed, failed)`. A row with `skipped=True` prints SKIP and counts
    toward neither.
    """
    cols = [
        ("scenario", 30, "scenario"),
        ("status", 7, "ok"),
        ("elapsed", 9, "elapsed_s"),
        ("notes", 90, "notes"),
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
                v = f"{v:.1f}s"
            cells.append(f"{str(v):<{w}}"[:w])
        print("  ".join(cells))
    print("=" * len(header))
    tail = f"  SKIPPED: {skipped}" if skipped else ""
    print(f"PASSED: {passed}  FAILED: {failed}  TOTAL: {passed + failed}{tail}")
    return passed, failed


def select_scenarios(all_names: list[str], requested: list[str] | None) -> list[str]:
    """Resolve repeatable/comma-separated `--scenario` filters against the battery.

    An unknown name is a hard error: a typo'd filter that quietly runs nothing
    reads as a clean pass.
    """
    if not requested:
        return all_names
    want = [s.strip() for item in requested for s in item.split(",") if s.strip()]
    unknown = [w for w in want if w not in all_names]
    if unknown:
        raise SystemExit(f"unknown scenario(s) {unknown}. Available: {all_names}")
    return want
