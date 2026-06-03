"""Shared helpers for the e2e SageMaker Flux correctness tests.

The Flux e2e driver (`e2e_test_streaming.py`) uses this to:

- Download the canonical Deepgram sample (`https://dpgr.am/spacewalk.wav`,
  ~25 s English mono) and verify it is 16-bit PCM.
- Multiply it by N loops in-place (concat raw frames inside a fresh WAV header)
  to a target duration — long enough to exercise sustained-concurrency
  behavior (default ~15 min).
- Compute a simple word-level Word Error Rate (WER) against the known
  reference transcript, with case + punctuation normalization. WER is the
  pass/fail signal across every scenario.

Flux is turn-based (`/v2/listen`): the authoritative final text for a turn is
the `EndOfTurn` transcript. The driver concatenates every `EndOfTurn`
transcript a connection produced into one `combined_final_text` and scores
that against the reference — so WER measures whole-stream content correctness
the same way it does for the Nova streaming driver, despite the different
protocol.

There are no AWS dependencies in this module — it is import-safe from any
context (cron, unit test, …).
"""

from __future__ import annotations

import math
import re
import urllib.request
import wave
from pathlib import Path

SPACEWALK_URL = "https://dpgr.am/spacewalk.wav"

# The canonical reference transcript for `spacewalk.wav`. Verbatim from the
# Deepgram demo page; one filler-word-rich utterance, ~25 s long.
SPACEWALK_REFERENCE_TEXT = (
    "Yeah, as much as um it's worth celebrating uh the uh first spacewalk "
    "with an all-female team, I think many of us uh are looking forward to "
    "it just being normal. And um I think if it signifies anything, it is "
    "uh to honor the the women who came before us who um were skilled and "
    "qualified um and didn't get uh the same opportunities that we have today."
)


# ---------------------------------------------------------------------------
# Fixture management
# ---------------------------------------------------------------------------

def download_sample(dst: Path, url: str = SPACEWALK_URL, *, force: bool = False) -> Path:
    """Download the sample WAV to `dst` (idempotent unless `force=True`)."""
    if dst.exists() and not force:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp:
        dst.write_bytes(resp.read())
    return dst


def validate_pcm16(wav_path: Path) -> tuple[int, int, float]:
    """Verify the file is 16-bit PCM and return (sample_rate, channels, duration_s)."""
    with wave.open(str(wav_path), "rb") as wf:
        sample_width = wf.getsampwidth()
        if sample_width != 2:
            raise ValueError(
                f"{wav_path} must be 16-bit PCM (sample width 2 bytes). "
                f"Got {sample_width * 8}-bit. Convert with: "
                "ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 output.wav"
            )
        sr = wf.getframerate()
        ch = wf.getnchannels()
        n_frames = wf.getnframes()
    return sr, ch, n_frames / sr


def multiply_wav(src: Path, dst: Path, target_seconds: float) -> int:
    """Loop `src` until `dst` is at least `target_seconds` long. Returns loop count.

    The output WAV reuses the source header (sample rate / channels / sample
    width). Frames are written as concatenated raw bytes — no resampling, no
    fades; seam clicks are part of the test (looped audio is what we have).
    """
    with wave.open(str(src), "rb") as wf:
        params = wf.getparams()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)
    src_dur = n_frames / sr
    loops = max(1, math.ceil(target_seconds / src_dur))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as wf:
        wf.setparams(params)
        for _ in range(loops):
            wf.writeframes(frames)
    return loops


def expected_text_for_loops(loops: int, *, sep: str = " ") -> str:
    """Reference transcript for an N-loop multiplied spacewalk.wav."""
    return sep.join([SPACEWALK_REFERENCE_TEXT] * loops)


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------

_NORMALIZE_STRIP = re.compile(r"[^\w\s']")
_NORMALIZE_SPACES = re.compile(r"\s+")

# Filler tokens the model strips at default settings. WER is computed after
# these are removed from BOTH ref and hyp so the metric measures content
# correctness rather than disfluency-marking.
FILLER_TOKENS = frozenset({
    "um", "umm", "ummm",
    "uh", "uhh", "uhhh",
    "ah", "ahh",
    "er", "err",
    "hm", "hmm", "hmmm",
    "mm", "mmm",
    "mhm", "mmhmm", "mmhm",
    "uhhuh", "uhuh",
})


def normalize_for_wer(text: str) -> list[str]:
    """Lowercase, strip non-word/non-apostrophe chars, drop filler tokens,
    tokenize on whitespace.

    Apostrophes are kept (so "it's" stays one token, matching ASR output).
    Fillers (see ``FILLER_TOKENS``) are dropped on both ref + hyp sides.
    """
    if not text:
        return []
    t = text.lower()
    t = _NORMALIZE_STRIP.sub(" ", t)
    t = _NORMALIZE_SPACES.sub(" ", t).strip()
    if not t:
        return []
    return [tok for tok in t.split() if tok not in FILLER_TOKENS]


def wer(reference: str, hypothesis: str) -> tuple[float, int, int, int, int]:
    """Word Error Rate via token-level Levenshtein.

    Returns `(wer_ratio, substitutions, deletions, insertions, ref_word_count)`.
    `wer_ratio == 1.0` is a sentinel for "nothing to score against" (empty ref).
    """
    r = normalize_for_wer(reference)
    h = normalize_for_wer(hypothesis)
    if not r:
        return (1.0 if h else 0.0, 0, 0, len(h), 0)

    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    op = [[" "] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
        op[i][0] = "d"
    for j in range(len(h) + 1):
        dp[0][j] = j
        op[0][j] = "i"
    op[0][0] = " "
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "="
            else:
                sub = dp[i - 1][j - 1] + 1
                dele = dp[i - 1][j] + 1
                ins = dp[i][j - 1] + 1
                best = min(sub, dele, ins)
                dp[i][j] = best
                op[i][j] = "s" if best == sub else ("d" if best == dele else "i")

    s = d = ins = 0
    i, j = len(r), len(h)
    while i > 0 or j > 0:
        o = op[i][j]
        if o == "=":
            i -= 1; j -= 1
        elif o == "s":
            s += 1; i -= 1; j -= 1
        elif o == "d":
            d += 1; i -= 1
        else:  # "i"
            ins += 1; j -= 1
    return (dp[len(r)][len(h)] / len(r), s, d, ins, len(r))


def fmt_wer(ratio: float) -> str:
    return f"{ratio * 100:.2f}%"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary_table(rows: list[dict], wer_threshold: float = 0.05) -> tuple[int, int]:
    """Render the per-scenario summary table. Returns (pass_count, fail_count)."""
    if not rows:
        print("(no scenarios ran)")
        return 0, 0

    cols = [
        ("scenario",         30, "scenario"),
        ("status",            7, "ok"),
        ("WER",               9, "wer"),
        ("S/D/I",            13, "sdi"),
        ("words",             8, "words"),
        ("elapsed",          10, "elapsed_s"),
        ("notes",            48, "notes"),
    ]
    header = "  ".join(f"{title:<{w}}" for title, w, _ in cols)
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    passed = failed = 0
    for r in rows:
        ok = r.get("ok")
        if ok is None:
            ok = r.get("wer", 1.0) <= wer_threshold and not r.get("error")
        if ok:
            passed += 1
        else:
            failed += 1
        cells = []
        for _, w, key in cols:
            v = r.get(key, "")
            if key == "ok":
                v = "PASS" if ok else "FAIL"
            elif key == "wer" and isinstance(v, (float, int)):
                v = fmt_wer(v)
            elif key == "elapsed_s" and isinstance(v, (float, int)):
                v = f"{v:.2f}s"
            elif key == "sdi" and isinstance(v, tuple) and len(v) == 3:
                v = f"{v[0]}/{v[1]}/{v[2]}"
            else:
                v = str(v) if v is not None else ""
            cells.append(f"{v:<{w}}")
        print("  ".join(cells))
    print("=" * len(header))
    print(f"PASSED: {passed}  FAILED: {failed}  TOTAL: {len(rows)}")
    return passed, failed
