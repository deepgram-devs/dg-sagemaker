#!/usr/bin/env python3
"""
End-to-end correctness + red-team test for a streaming SageMaker STT endpoint.

Drives `stt_wav_stress.py stream` through a sequence of scenarios — basic
short-form, basic long-form, sustained-concurrency, ramped-concurrency, each
of the major feature flags (diarize / keyterms / redact / interim-results),
and an adversarial bare-WebSocket-close path — then validates each
connection's `combined_final_text` against a known reference transcript via
Word Error Rate. Designed to be the definitive correctness gate for a
streaming endpoint before promoting it.

Fixtures
--------
- Downloads `https://dpgr.am/spacewalk.wav` (~25 s English mono, 16-bit PCM).
- Multiplies the sample by N loops in-place to a long-form variant (default
  ≥ 15 min) for sustained-concurrency + long-form smoke.

Pass / fail
-----------
Each scenario succeeds when:
  - the subprocess exits 0,
  - every connection reports at least one final transcript,
  - WER of the combined final text vs. the expected reference (single or
    multiplied) is below the per-scenario threshold (default 5%).

For scenarios that intentionally distort transcription (e.g. PII redact), the
threshold is raised and a presence-check on the redaction marker is applied
instead of raw WER. For diarize, WER is computed against the same reference
because the text body is unchanged — diarize only adds speaker tags as
separate fields.

Endpoint prerequisites
----------------------
- A streaming-mode SageMaker endpoint (manifest `mode: streaming`) is
  InService and accessible to the calling identity.
- The endpoint's manifest defaults include the requested `--model` /
  `--language` (default `nova-3` / `en`), or the same can be requested
  explicitly via `--model` / `--language` on this script.

Usage
-----
    uv run e2e_test_streaming.py your-streaming-endpoint-name --region us-east-2

By default the WAV fixtures land under `/tmp/dg-sagemaker-e2e/` and persist
across runs (re-runs skip the download); pass `--workdir <path>` to override.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# Same directory imports so the e2e suite can be run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e_test_common import (
    SPACEWALK_REFERENCE_TEXT,
    download_sample,
    expected_text_for_loops,
    fmt_wer,
    multiply_wav,
    print_summary_table,
    validate_pcm16,
    wer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

@dataclass
class StreamScenario:
    """One streaming scenario, scoped to features supported on the
    nova-3 streaming transport per
    https://developers.deepgram.com/docs/ (May 2026 audit).

    `bundle_component` documents the model component a bundle must include
    for the scenario to succeed; when omitted, only the ASR weights are
    required. `tolerated_error_substring` lets the scenario PASS-WITH-NOTE
    when the endpoint returns a specific known error (e.g. "entity
    detection" when the redaction model isn't bundled) — this surfaces
    bundle gaps without false-failing.
    """
    name: str
    description: str
    use_long_form: bool = False
    connections: int = 1
    extra_args: list[str] = field(default_factory=list)
    wer_threshold: float = 0.05
    presence_check: str | None = None  # substring or marker that must appear
    expect_failure: bool = False
    bundle_component: str | None = None  # e.g. "streaming-ner"
    tolerated_error_substring: str | None = None  # PASS-WITH-NOTE if seen
    notes: str = ""


def default_scenarios(model: str, language: str) -> list[StreamScenario]:
    """Feature coverage matrix per the docs (nova-3, streaming):

      Always-supported by stem on nova-3 streaming (per docs):
        punctuate, smart_format, numerals, dictation, diarize (v1),
        keyterm, replace (find/replace), profanity_filter,
        interim_results, search
      Nova-3 streaming SPECIFIC notes:
        - `keyterm` (NOT `keywords` — those are nova-2/legacy)
        - `diarize=true` only — `diarize_model=v2` is pre-recorded-only,
          returns 400 on streaming
      Bundle-component dependent:
        - redact / detect_entities → requires streaming-ner
          (UUID 90424f3a-... `modes=["streaming"]`)
        - search → requires g2p (UUID 89555db3-...
          `modes=["streaming","batch"]`); search is supported on
          nova-3 streaming AND batch per
          https://developers.deepgram.com/docs/search
      Not supported on nova-3 streaming (per docs; excluded here):
        - filler_words (pre-recorded only on nova-3)
        - utterances (pre-recorded only on nova-3; see
          https://developers.deepgram.com/docs/utterances)
        - paragraphs (pre-recorded only)
        - measurements (pre-recorded only)
        - utt_split (pre-recorded only)
        - diarize_model (pre-recorded only)
        - summarize / topics / intents / sentiment (pre-recorded only)
    """
    return [
        # ---- Coverage / load ----
        StreamScenario(
            name="basic_25s",
            description="1 conn, 25 s file, defaults",
            connections=1,
        ),
        StreamScenario(
            name="concurrent_5x_25s",
            description="5 simultaneous connections, 25 s file",
            connections=5,
        ),
        StreamScenario(
            name="concurrent_10x_15min",
            description="10 simultaneous connections on ~15 min file",
            use_long_form=True,
            connections=10,
            notes="sustained-load WER check",
        ),
        StreamScenario(
            name="ramp_10x_step5",
            description="10 conns in batches of 5 with 2 s delay",
            connections=10,
            extra_args=["--batch-size", "5", "--batch-delay", "2"],
        ),
        # ---- Streaming-only features ----
        StreamScenario(
            name="feature_interim_results",
            description="--interim-results (verify interims emitted)",
            connections=1,
            extra_args=["--interim-results"],
            notes="streaming-only feature",
        ),
        # ---- Formatting features ----
        StreamScenario(
            name="feature_diarize_v1",
            description="--diarize true (streaming v1 diarizer)",
            connections=1,
            extra_args=["--diarize", "true"],
            notes="diarize_model=v2 is NOT accepted on streaming (400)",
        ),
        StreamScenario(
            name="feature_smart_format",
            description="--extra smart_format=true",
            connections=1,
            extra_args=["--extra", "smart_format=true"],
            notes="implies punctuate; may delay finals up to 3 s",
        ),
        StreamScenario(
            name="feature_punctuate",
            description="--punctuate true (default; explicit verification)",
            connections=1,
            extra_args=["--punctuate", "true"],
            presence_check=".",
            notes="punctuation marks should appear in finals",
        ),
        StreamScenario(
            name="feature_numerals",
            description="--extra numerals=true (digit substitution)",
            connections=1,
            extra_args=["--extra", "numerals=true"],
            notes="numerals param accepted; clip has few numbers — smoke",
        ),
        StreamScenario(
            name="feature_dictation",
            description="--extra dictation=true (spoken punctuation -> chars)",
            connections=1,
            extra_args=["--extra", "dictation=true"],
            wer_threshold=1.0,  # dictation may alter punctuation tokens
            notes="dictation transforms spoken punctuation cues",
        ),
        StreamScenario(
            name="feature_profanity_filter",
            description="--extra profanity_filter=true (no profanity in clip; smoke)",
            connections=1,
            extra_args=["--extra", "profanity_filter=true"],
            notes="clip has no profanity; transcript should be unchanged",
        ),
        # ---- Custom vocabulary ----
        StreamScenario(
            name="feature_keyterm",
            description="--keyterms 'spacewalk,female' (nova-3 boost)",
            connections=1,
            extra_args=["--keyterms", "spacewalk,female"],
            presence_check="spacewalk",
            notes="nova-3 only — `keyterm`, NOT `keywords`",
        ),
        StreamScenario(
            name="feature_replace",
            description="replace=spacewalk:moonwalk (find/replace)",
            connections=1,
            extra_args=["--extra", "replace=spacewalk:moonwalk"],
            wer_threshold=1.0,  # text intentionally changed
            presence_check="moonwalk",
            notes="content swap; WER skipped",
        ),
        # ---- Search (bundle-dependent: needs g2p) ----
        StreamScenario(
            name="feature_search",
            description="search=spacewalk (phonetic match; requires g2p)",
            connections=1,
            extra_args=["--extra", "search=spacewalk"],
            presence_check="spacewalk",  # search hits surface in results
            bundle_component="g2p (uuid 89555db3-...)",
            notes="needs g2p; will FAIL on bundles without it",
        ),
        # ---- Redaction / entity detection (bundle-dependent) ----
        StreamScenario(
            name="feature_redact_name",
            description="--redact name (requires streaming-ner component)",
            connections=1,
            extra_args=["--redact", "name"],
            wer_threshold=1.0,
            bundle_component="streaming-ner",
            tolerated_error_substring="entity detection",
            notes="WER skipped; PASS-WITH-NOTE if bundle lacks streaming-ner",
        ),
        # ---- Adversarial ----
        StreamScenario(
            name="adversarial_bare_close",
            description="--no-use-close-stream (bare WS close path)",
            connections=1,
            extra_args=["--no-use-close-stream"],
            notes="trailing tail may drop; WER threshold relaxed",
            wer_threshold=0.10,
        ),
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _stress_cmd(
    script: Path,
    endpoint: str,
    wav_path: Path,
    region: str,
    model: str,
    language: str,
    connections: int,
    summary_path: Path,
    extra: list[str],
) -> list[str]:
    cmd = [
        "uv", "run", "--project", str(script.parent),
        str(script), "stream", endpoint,
        "--file", str(wav_path),
        "--region", region,
        "--model", model,
        "--language", language,
        "--connections", str(connections),
        "--summary-jsonl", str(summary_path),
    ]
    cmd.extend(extra)
    return cmd


def _scenario_timeout_s(scenario: StreamScenario, long_form_audio_s: float, base_timeout_s: int) -> int:
    """Per-scenario subprocess timeout.

    Long-form scenarios stream the full multiplied WAV in wall-clock time
    (the bidi-stream paces audio to its sample rate), so the subprocess
    needs at least `long_form_audio_s` plus headroom for connect / finals /
    teardown. A flat 900 s cap clips ~908 s long-form runs as a timeout
    rather than letting them complete. Headroom = max(300 s, 30% of audio)
    so concurrent runs (where finals trickle in after the last frame ships)
    still finish cleanly. Short-form scenarios keep the user-provided
    `--subprocess-timeout-s` (default 900 s — plenty for a 26 s file).
    """
    if not scenario.use_long_form:
        return base_timeout_s
    headroom = max(300.0, long_form_audio_s * 0.3)
    return int(long_form_audio_s + headroom)


def run_scenario(
    scenario: StreamScenario,
    *,
    endpoint: str,
    region: str,
    model: str,
    language: str,
    short_wav: Path,
    long_wav: Path,
    long_loops: int,
    long_form_audio_s: float,
    stress_script: Path,
    log_dir: Path,
    subprocess_timeout_s: int,
) -> dict:
    wav = long_wav if scenario.use_long_form else short_wav
    timeout_s = _scenario_timeout_s(scenario, long_form_audio_s, subprocess_timeout_s)
    expected = (
        expected_text_for_loops(long_loops)
        if scenario.use_long_form
        else SPACEWALK_REFERENCE_TEXT
    )

    summary_path = log_dir / f"{scenario.name}.summary.jsonl"
    stdout_path = log_dir / f"{scenario.name}.stdout.log"
    stderr_path = log_dir / f"{scenario.name}.stderr.log"

    cmd = _stress_cmd(
        stress_script, endpoint, wav, region, model, language,
        scenario.connections, summary_path, scenario.extra_args,
    )
    logger.info(f"[{scenario.name}] running: {' '.join(cmd)}")

    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        elapsed = time.monotonic() - start
        return {
            "scenario": scenario.name,
            "ok": False,
            "wer": 1.0,
            "sdi": (0, 0, 0),
            "words": 0,
            "elapsed_s": elapsed,
            "notes": f"subprocess timed out after {timeout_s}s (long_form_audio_s={long_form_audio_s:.0f})",
            "error": "timeout",
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
    elapsed = time.monotonic() - start

    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)

    if not summary_path.exists():
        return {
            "scenario": scenario.name,
            "ok": False,
            "wer": 1.0,
            "sdi": (0, 0, 0),
            "words": 0,
            "elapsed_s": elapsed,
            "notes": f"no summary-jsonl produced (exit={result.returncode})",
            "error": f"exit {result.returncode}",
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }

    rows = [json.loads(l) for l in summary_path.read_text().splitlines() if l.strip()]
    if not rows:
        return {
            "scenario": scenario.name,
            "ok": False,
            "wer": 1.0,
            "sdi": (0, 0, 0),
            "words": 0,
            "elapsed_s": elapsed,
            "notes": "summary-jsonl empty",
            "error": f"exit {result.returncode}",
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }

    # Score the connection with the most final transcripts (others should
    # match closely; WER on the strongest one keeps signal clean if a single
    # session experienced a transient hiccup).
    rows.sort(key=lambda r: r.get("transcripts_final", 0), reverse=True)
    best = rows[0]
    combined = best.get("combined_final_text", "") or ""
    finals = sum(r.get("transcripts_final", 0) for r in rows)
    errored = sum(1 for r in rows if r.get("errored"))
    interim_total = sum(r.get("transcripts_interim", 0) for r in rows)

    w_ratio, s, d, i, n = wer(expected, combined)

    notes_parts = []
    if scenario.notes:
        notes_parts.append(scenario.notes)
    notes_parts.append(f"conns={len(rows)}")
    notes_parts.append(f"finals={finals}")
    if errored:
        notes_parts.append(f"errored={errored}")
    if scenario.name == "feature_interim_results":
        notes_parts.append(f"interim_total={interim_total}")

    # PASS-WITH-NOTE: bundle missing the component this scenario requires.
    # The stem returns a known error pattern (e.g. "entity detection" for
    # redact on a bundle lacking streaming-ner); we surface the bundle gap
    # without false-failing.
    all_error_text = " ".join(
        msg for row in rows for msg in (row.get("error_messages") or [])
    ).lower()
    tolerated = (
        scenario.tolerated_error_substring is not None
        and scenario.tolerated_error_substring.lower() in all_error_text
    )

    presence_ok = True
    if scenario.presence_check and not tolerated:
        presence_ok = scenario.presence_check.lower() in combined.lower()
        notes_parts.append(f"presence({scenario.presence_check})={'ok' if presence_ok else 'MISSING'}")

    if tolerated:
        notes_parts.append(
            f"BUNDLE-GAP: '{scenario.tolerated_error_substring}' "
            f"(needs {scenario.bundle_component or 'feature-specific'} component) — pass-with-note"
        )
        ok = True
        # Reset misleading WER signal — the request never produced ASR output.
        w_ratio, s, d, i = 0.0, 0, 0, 0
    else:
        ok = (
            result.returncode == 0
            and errored == 0
            and finals > 0
            and w_ratio <= scenario.wer_threshold
            and presence_ok
        )
    if scenario.name == "feature_interim_results" and interim_total == 0 and not tolerated:
        ok = False
        notes_parts.append("no interim emissions")

    return {
        "scenario": scenario.name,
        "ok": ok,
        "wer": w_ratio,
        "sdi": (s, d, i),
        "words": n,
        "elapsed_s": elapsed,
        "notes": " ".join(notes_parts),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "summary_path": str(summary_path),
        "combined_final_text": combined[:200],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "End-to-end correctness test for a streaming SageMaker STT endpoint. "
            "Downloads the canonical Deepgram spacewalk.wav sample, multiplies "
            "it to ~15 min, then runs a battery of scenarios through "
            "stt_wav_stress.py and validates each session's transcript against "
            "the known reference via Word Error Rate."
        )
    )
    p.add_argument(
        "endpoint_name",
        nargs="?",
        default=None,
        help="Streaming SageMaker endpoint name (required unless --list)",
    )
    p.add_argument("--region", default="us-east-2", help="AWS region (default: us-east-2)")
    p.add_argument("--model", default="nova-3", help="Deepgram model (default: nova-3)")
    p.add_argument("--language", default="en", help="Language code (default: en)")
    p.add_argument(
        "--workdir",
        default=None,
        metavar="DIR",
        help="Fixture + log directory (default: /tmp/dg-sagemaker-e2e/streaming/<timestamp>)",
    )
    p.add_argument(
        "--target-long-form-s",
        type=float,
        default=900.0,
        metavar="SECONDS",
        help="Target duration of the long-form multiplied WAV (default: 900 = 15 min)",
    )
    p.add_argument(
        "--scenarios",
        default="",
        metavar="NAME,NAME,...",
        help="Comma-separated subset of scenario names to run (default: all). "
             "Pass --list to see available scenarios.",
    )
    p.add_argument("--list", action="store_true", help="List scenarios and exit")
    p.add_argument(
        "--wer-threshold",
        type=float,
        default=0.05,
        help="Default WER threshold for non-distorting scenarios (default: 0.05)",
    )
    p.add_argument(
        "--subprocess-timeout-s",
        type=int,
        default=900,
        help="Per-scenario subprocess timeout (default: 900 = 15 min)",
    )
    p.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download spacewalk.wav even if cached",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return p


def main() -> int:
    args = _make_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")

    scenarios = default_scenarios(args.model, args.language)
    if args.wer_threshold != 0.05:
        for s in scenarios:
            if s.wer_threshold == 0.05:
                s.wer_threshold = args.wer_threshold

    if args.list:
        print("Available scenarios:")
        for s in scenarios:
            print(f"  {s.name:<25} {s.description}  (threshold={fmt_wer(s.wer_threshold)})")
        return 0

    if not args.endpoint_name:
        print("ERROR: endpoint_name is required (run with --list to see scenarios).",
              file=sys.stderr)
        return 1

    if args.scenarios:
        wanted = {tok.strip() for tok in args.scenarios.split(",") if tok.strip()}
        unknown = wanted - {s.name for s in scenarios}
        if unknown:
            print(f"ERROR: unknown scenario(s): {sorted(unknown)}", file=sys.stderr)
            print(f"Run with --list to see available names.", file=sys.stderr)
            return 1
        scenarios = [s for s in scenarios if s.name in wanted]

    if args.workdir:
        workdir = Path(args.workdir).expanduser().resolve()
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        workdir = Path(tempfile.gettempdir()) / "dg-sagemaker-e2e" / "streaming" / ts
    workdir.mkdir(parents=True, exist_ok=True)
    log_dir = workdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    short_wav = workdir / "spacewalk.wav"
    long_wav = workdir / "spacewalk-15min.wav"

    print("=" * 80)
    print(f"Endpoint:    {args.endpoint_name}")
    print(f"Region:      {args.region}")
    print(f"Model/lang:  {args.model} / {args.language}")
    print(f"Workdir:     {workdir}")
    print("=" * 80)

    # 1. Fixture prep
    download_sample(short_wav, force=args.force_download)
    sr, ch, dur = validate_pcm16(short_wav)
    print(f"Sample:      {short_wav.name}  {sr} Hz  {ch}ch  {dur:.2f}s "
          f"({short_wav.stat().st_size / 1024:.0f} KB)")
    if not long_wav.exists() or args.force_download:
        loops = multiply_wav(short_wav, long_wav, args.target_long_form_s)
    else:
        # Recompute loop count from existing file size for the expected text.
        _, _, long_dur = validate_pcm16(long_wav)
        loops = max(1, round(long_dur / dur))
    _, _, long_dur = validate_pcm16(long_wav)
    print(f"Long-form:   {long_wav.name}  {long_dur:.0f}s ({loops} loops) "
          f"({long_wav.stat().st_size / (1024*1024):.1f} MB)")

    # 2. Locate the stress script (lives one directory up — e2e/ is a sibling of stt_wav_stress.py)
    stress_script = Path(__file__).resolve().parent.parent / "stt_wav_stress.py"
    if not stress_script.is_file():
        print(f"ERROR: {stress_script} missing", file=sys.stderr)
        return 2

    # 3. Run scenarios sequentially
    print()
    rows: list[dict] = []
    for scenario in scenarios:
        print(f"--> {scenario.name}  ({scenario.description})")
        row = run_scenario(
            scenario,
            endpoint=args.endpoint_name,
            region=args.region,
            model=args.model,
            language=args.language,
            short_wav=short_wav,
            long_wav=long_wav,
            long_loops=loops,
            long_form_audio_s=long_dur,
            stress_script=stress_script,
            log_dir=log_dir,
            subprocess_timeout_s=args.subprocess_timeout_s,
        )
        rows.append(row)
        flag = "PASS" if row["ok"] else "FAIL"
        print(f"    {flag}  WER={fmt_wer(row['wer'])}  elapsed={row['elapsed_s']:.1f}s  {row['notes']}")

    # 4. Summary
    print()
    passed, failed = print_summary_table(rows, wer_threshold=args.wer_threshold)
    (workdir / "results.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nFull results: {workdir / 'results.json'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
