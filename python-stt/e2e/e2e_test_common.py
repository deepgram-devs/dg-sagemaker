"""Shared helpers for the e2e SageMaker correctness tests.

The two e2e drivers (`e2e_test_streaming.py`, `e2e_test_batch.py`) use this to:

- Download the canonical Deepgram sample (`https://dpgr.am/spacewalk.wav`,
  ~25 s English mono) and verify it is 16-bit PCM.
- Multiply it by N loops in-place (concat raw frames inside a fresh WAV header)
  to a target duration — long enough to exercise long-form behavior (default
  ~15 min, well beyond the 25 MB sync InvokeEndpoint body limit).
- Compute a simple word-level Word Error Rate (WER) against the known
  reference transcript, with case + punctuation normalization. WER is the
  pass/fail signal across every scenario: ASR for this sample should land
  comfortably under 5 % on a healthy endpoint; high WER (or any decode that
  collapses to an empty string) indicates a regression worth surfacing.

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
# Japanese fixture (COMMITTED, not downloaded)
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS. Every other clip in this suite is English, which makes the
# east-asian nova-3 bundles (zh/ja/ko/vi/id/th) unverifiable: they carry no `en`
# model, so `--language en` is rejected outright, and English audio under
# `--language ja` is ACCEPTED and returns nothing. Measured 2026-09-11 on the
# published eastasia streaming version: every connection lived its full ~37 s
# and returned `finals=0`, which is indistinguishable from a model that is
# simply broken. A release was signed off on liveness alone because of it.
#
# Synthesised with Deepgram TTS so it is ours to commit (no third-party audio
# rights question) and reproducible:
#
#   curl -X POST "https://api.deepgram.com/v1/speak\
#       ?model=aura-2-fujin-ja&encoding=linear16&sample_rate=24000" \
#     -H "Authorization: Token $DEEPGRAM_API_KEY" \
#     -H "Content-Type: application/json" \
#     -d '{"text": "<JA_REFERENCE_TEXT>"}'
#
# NOTE the response's RIFF header carries a streaming placeholder size
# (0x7fff0024), so the bytes must be re-wrapped with correct chunk sizes before
# `wave` reports a usable frame count — the committed file already is.
#
# `aura-2-fujin-ja` was chosen over `aura-2-izanami-ja` by round-tripping both
# through nova-3 `language=ja`: fujin came back at confidence 1.0 retaining
# 宇宙飛行士, where izanami dropped 宇宙. Mono / 24 kHz / 16-bit / ~22.5 s,
# RMS 1171, peak 15856.
JA_FIXTURE = Path(__file__).parent / "fixtures" / "ja_reference_24k.wav"

JA_REFERENCE_TEXT = (
    "こんにちは。これは日本語の音声認識をテストするための音声です。"
    "今日の東京の天気は晴れで、気温は二十度です。"
    "会議は午後三時に始まりますので、資料を準備してください。"
    "宇宙飛行士のチームが来週、記者会見を行う予定です。"
    "ご不明な点がありましたら、遠慮なくお問い合わせください。"
    "よろしくお願いいたします。"
)

# WORD-LEVEL WER IS UNUSABLE ON THIS CLIP — use `cer()` instead.
#
# Be precise about why, because the obvious explanation is wrong. Japanese has
# no spaces, but `normalize_for_wer` replaces punctuation with spaces, so the
# text DOES tokenise — at clause boundaries, not word ones. Measured
# 2026-09-11 against a real nova-3 `language=ja` transcript of this very file:
# 3 wrong characters out of 135 scored CER 2.22% and WER 36.36%, because each
# "word" is an entire clause and one bad character fails the whole thing.
#
# So WER here is not binary (an earlier note in this file claimed it was, from
# a raw `.split()` measurement that the driver does not use) — it is simply
# wildly pessimistic and not comparable to an English WER figure. Judge this
# fixture on `cer()` plus the presence terms below.
#
# Substrings a working `ja` model must produce. Presence of these + finals > 0
# is the real signal, and it is exactly the signal the English clip cannot give
# for these bundles.
#
# EVERY TERM HERE MUST BE NUMERAL-FREE. `午後三時` was in this list initially
# and had to come out: the hosted API returned 午後三時 and 二十度, while the
# SageMaker eastasia bundle returned 午後3時 and 20度 for the same audio
# (measured 2026-09-11 on ja-ea-strm). Digit-vs-kanji rendering is a formatting
# choice, not a health signal, so a term that depends on it fails a perfectly
# good model. Verified present on BOTH surfaces.
# CER gate for the ja fixture. The English scenarios use a 5% WER threshold and
# CER MUST NOT inherit it — they measure different things on different units,
# and wiring the metric in without its own gate failed a healthy endpoint on
# 2026-09-11 (CER 7.41% judged against 5.00%).
#
# Measured baseline, published eastasia streaming bundle on ml.g5.2xlarge:
# CER 7.41% (4 subs / 6 dels of 135 chars), IDENTICAL on all five content
# scenarios and at both 1 and 5 concurrent connections — deterministic, not a
# noisy sample. 15% leaves roughly 2x headroom for a slower instance family or
# a different bundle while still failing a transcript that has collapsed
# (empty output scores 100%).
#
# LIKE-FOR-LIKE REFERENCE (measured 2026-09-11, same clip, same query params,
# same CHUNK_SIZE=8192 real-time pacing, finals concatenated the same way):
#
#   hosted   wss://api.deepgram.com  streaming   CER 10.37%  (11 finals, 14 del)
#   SageMaker  eastasia buyer endpoint           CER  7.41%  (12 finals, 4s/6d)
#   hosted   batch REST                         CER  2.22%  ( 3 del)
#
# So SageMaker streaming BEATS hosted streaming on this clip, and the big gap
# is transport (batch has full context; streaming finalises incrementally) —
# not a SageMaker deficiency. An earlier version of this comment compared
# SageMaker streaming against hosted BATCH and implied a ~5-point shortfall;
# that comparison was not like-for-like and the conclusion was wrong.
#
# 15% therefore clears both streaming surfaces with the worse of the two
# (hosted, 10.37%) still inside it. Track movement in e2e-results rather than
# tightening on one endpoint's reading.
JA_CER_THRESHOLD = 0.15

JA_PRESENCE_TERMS = ("東京", "宇宙飛行士", "記者会見", "音声認識")


def ja_fixture() -> Path:
    """Path to the committed Japanese clip; raises if it is missing.

    Unlike `download_sample`, this does NOT fetch anything — the point of
    committing it is that a target-language verification cannot silently
    degrade into "no audio available, skipped".
    """
    if not JA_FIXTURE.exists():
        raise FileNotFoundError(
            f"Japanese fixture missing at {JA_FIXTURE}. It is committed to the "
            "repo on purpose; restore it rather than skipping the check."
        )
    return JA_FIXTURE


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
    width) so SageMaker pre-recorded mode treats it identically to the source.
    Frames are written as concatenated raw bytes — no resampling, no fades,
    seam clicks are part of the test (looped audio is what we have).
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


def trim_trailing_silence(
    src: Path, dst: Path, *, threshold_frac: float = 0.015, min_keep_s: float = 1.0
) -> float:
    """Write `dst` = `src` with trailing (near-)silence removed, so the clip ends
    right at the last speech sample. Returns the trimmed duration in seconds.

    This recreates the "pre-segmented telephony, little/no trailing silence"
    condition: with endpointing enabled, the final segment never
    sees the trailing silence that would endpoint it in-stream, so it is emitted
    ONLY if the server's CloseStream-triggered finalize is delivered. A clip with
    trailing silence (like raw spacewalk.wav) endpoints its last segment normally
    and therefore can't catch a tail-finalize regression. Speech content is
    untouched, so the reference transcript is unchanged.
    """
    import array

    with wave.open(str(src), "rb") as wf:
        sw, sr, ch = wf.getsampwidth(), wf.getframerate(), wf.getnchannels()
        if sw != 2:
            raise ValueError(f"{src} must be 16-bit PCM")
        samples = array.array("h")
        samples.frombytes(wf.readframes(wf.getnframes()))

    total_frames = len(samples) // ch
    peak = max((abs(s) for s in samples), default=1) or 1
    thr = peak * threshold_frac
    # Scan backward for the last frame whose loudest channel exceeds threshold.
    last_voiced = 0
    for f in range(total_frames - 1, -1, -1):
        base = f * ch
        if max(abs(samples[base + c]) for c in range(ch)) > thr:
            last_voiced = f
            break
    end_frame = max(last_voiced + 1, int(min_keep_s * sr))
    end_frame = min(end_frame, total_frames)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(sw)
        wf.setframerate(sr)
        wf.writeframes(samples[: end_frame * ch].tobytes())
    return end_frame / sr


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------

_NORMALIZE_STRIP = re.compile(r"[^\w\s']")
_NORMALIZE_SPACES = re.compile(r"\s+")

# Filler tokens nova-3 strips at its default settings (smart-format on).
# WER is computed after these are removed from BOTH ref and hyp so the metric
# measures content correctness rather than disfluency-marking — the reference
# transcript carries the speaker's literal fillers, but the ASR output doesn't,
# so an unfiltered WER would conflate "we got the words right" with "we
# preserved every um and uh".
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


_CER_STRIP = re.compile(r"[\s\u3000。、，．,\.!?！？「」『』（）\(\)・:：;；\-—ー~〜\"\']+")


def normalize_for_cer(text: str) -> str:
    """Collapse a transcript to a bare character sequence for CER.

    Strips whitespace (including U+3000 ideographic space) and the punctuation
    both sides disagree about cosmetically — Japanese ASR output varies on 、
    and 。 placement, and that is not an accuracy signal worth failing on.
    """
    return _CER_STRIP.sub("", text or "")


def cer(reference: str, hypothesis: str) -> tuple[float, int, int, int, int]:
    """Character Error Rate — the WER analogue for languages without spaces.

    Returns the same shape as `wer()`: `(ratio, subs, dels, ins, ref_len)`.

    WHY THIS IS REQUIRED, not a nicety. `wer()` tokenises on whitespace, and
    Japanese is written without spaces, so reference and hypothesis each
    collapse to exactly ONE token (measured 2026-09-11 on the committed ja
    fixture). The word-level gate can then only ever return 0% for an exact
    match or 100% for any difference at all, with nothing in between — it
    cannot express "mostly right". Use this for any CJK bundle.
    """
    r = normalize_for_cer(reference)
    h = normalize_for_cer(hypothesis)
    if not r:
        return (1.0 if h else 0.0, 0, 0, len(h), 0)
    # Levenshtein with operation tallies; same DP shape as wer() above.
    prev = [(j, 0, 0, j) for j in range(len(h) + 1)]
    for i_r, rc in enumerate(r, 1):
        cur = [(i_r, 0, i_r, 0)]
        for j_h, hc in enumerate(h, 1):
            if rc == hc:
                cost, sub, dele, ins = prev[j_h - 1]
                cur.append((cost, sub, dele, ins))
                continue
            sub_c = prev[j_h - 1][0] + 1
            del_c = prev[j_h][0] + 1
            ins_c = cur[j_h - 1][0] + 1
            best = min(sub_c, del_c, ins_c)
            if best == sub_c:
                _, a, b, c = prev[j_h - 1]; cur.append((best, a + 1, b, c))
            elif best == del_c:
                _, a, b, c = prev[j_h]; cur.append((best, a, b + 1, c))
            else:
                _, a, b, c = cur[j_h - 1]; cur.append((best, a, b, c + 1))
        prev = cur
    total, sub, dele, ins = prev[-1]
    return (total / len(r), sub, dele, ins, len(r))


def wer(reference: str, hypothesis: str) -> tuple[float, int, int, int, int]:
    """Word Error Rate via token-level Levenshtein.

    Returns `(wer_ratio, substitutions, deletions, insertions, ref_word_count)`.
    `wer_ratio == 1.0` is a sentinel for "nothing to score against" (empty ref).
    """
    r = normalize_for_wer(reference)
    h = normalize_for_wer(hypothesis)
    if not r:
        return (1.0 if h else 0.0, 0, 0, len(h), 0)

    # DP grid with operation tracking. Space O(2 * len(h)) would suffice but
    # the corpus here is small — keep it readable.
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
# Language-restricted scenarios
# ---------------------------------------------------------------------------

def language_supported(supported: list[str] | None, run_language: str) -> bool:
    """Does a scenario declaring `supported` (its `supported_languages` field)
    support the endpoint's configured `run_language`?

    `supported is None` means "no restriction" (every scenario's default) —
    only set `supported_languages` on a scenario when there's DIRECT evidence
    the endpoint hard-rejects the request outside that language set (e.g. a
    documented `400` with an explicit `"not supported for non-English
    languages"`-style body). Most "English" badges in the docs describe a
    feature that degrades gracefully (a `warnings[]` entry, or a silent
    no-op) rather than a hard reject — those scenarios should stay
    unrestricted so the graceful-degrade behavior itself keeps getting
    exercised. See the 2026-07-14 investigation of `summarize=v2` on a
    `language=multi` batch endpoint: initially assumed to be a bundle-staging
    gap, actually a hard, permanent `400` ("Summarization v2 not supported
    for non-English languages") — this predicate exists for exactly that
    class of scenario, not the softer "English only" cases.

    `run_language` matches an entry if exact, or if its base subtag matches
    (`en-US` matches an entry of `en`). `multi` only matches if `"multi"` is
    listed explicitly — a language-restricted scenario running against a
    language-detect endpoint should be treated as unsupported by default.
    """
    if supported is None:
        return True
    run = run_language.lower()
    return any(run == e.lower() or run.split("-")[0] == e.lower() for e in supported)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary_table(rows: list[dict], wer_threshold: float = 0.05) -> tuple[int, int]:
    """Render the per-scenario summary table. Returns (pass_count, fail_count).

    A row with `"skipped": True` (a scenario whose `supported_languages`
    excluded the run's `--language`) prints as SKIP and counts toward
    neither passed nor failed — the return tuple stays a 2-tuple so existing
    callers (`0 if failed == 0 else 1` exit codes) need no changes; the
    skipped count is still shown in the footer.
    """
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
        ("notes",            40, "notes"),
    ]
    header = "  ".join(f"{title:<{w}}" for title, w, _ in cols)
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    passed = failed = skipped = 0
    for r in rows:
        is_skip = bool(r.get("skipped"))
        ok = r.get("ok")
        if is_skip:
            skipped += 1
        else:
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
                v = "SKIP" if is_skip else ("PASS" if ok else "FAIL")
            elif key == "wer" and isinstance(v, (float, int)):
                v = "-" if is_skip else fmt_wer(v)
            elif key == "elapsed_s" and isinstance(v, (float, int)):
                v = f"{v:.2f}s"
            elif key == "sdi" and isinstance(v, tuple) and len(v) == 3:
                v = f"{v[0]}/{v[1]}/{v[2]}"
            else:
                v = str(v) if v is not None else ""
            cells.append(f"{v:<{w}}")
        print("  ".join(cells))
    print("=" * len(header))
    print(f"PASSED: {passed}  FAILED: {failed}  SKIPPED: {skipped}  TOTAL: {len(rows)}")
    return passed, failed
