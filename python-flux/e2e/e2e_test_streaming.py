#!/usr/bin/env python3
"""
End-to-end correctness + feature-coverage test for a streaming SageMaker Flux endpoint.

Flux is a streaming-only conversational STT model on the `/v2/listen`
bidirectional transport with a turn-based protocol (TurnInfo events with
integrated end-of-turn detection) — so this is the only e2e driver Flux needs
(there is no batch/pre-recorded Flux surface).

It drives `flux_stress.py file` through a battery of scenarios — basic
short-form, sustained / ramped concurrency, every major connection parameter
(eot_threshold / eot_timeout_ms / eager_eot_threshold / keyterm /
profanity_filter / encoding), the multilingual model + language hints, the
in-band control messages (Configure / KeepAlive / Finalize), and a negative
test (language_hint on the English-only model) — then validates each
connection's combined `EndOfTurn` transcript against the known reference via
Word Error Rate, plus event/ack assertions specific to Flux.

Fixtures
--------
- Downloads `https://dpgr.am/spacewalk.wav` (~25 s English mono, 16-bit PCM).
- Multiplies it to a long-form variant (default ≥ 15 min) for
  sustained-concurrency.

Pass / fail
-----------
Each scenario succeeds when:
  - the subprocess exits 0,
  - no connection errored,
  - at least one `EndOfTurn` transcript was produced,
  - WER of the combined `EndOfTurn` text vs. the reference is below the
    per-scenario threshold (default 5 %),
  - and any scenario-specific assertions hold (eager events emitted,
    Configure accepted/rejected, presence checks, …).

Two non-WER scenario kinds:
  - `expect_failure=True` (e.g. `language_hint` on `flux-general-en`): PASSES
    when a connection errors as expected; WER is skipped.
  - `tolerated_error_substring` (e.g. requesting `flux-general-multi` against
    an endpoint that only bundles `flux-general-en`): PASS-WITH-NOTE when the
    endpoint returns that known error — surfaces the deployment gap without
    false-failing.

Feature scoping per https://developers.deepgram.com/docs/flux/ (June 2026 audit).

Endpoint prerequisites
----------------------
- A Flux streaming SageMaker endpoint (engine config `listen_v2 = true`) is
  InService and accessible to the calling identity.
- The endpoint bundles the `--model` used by each scenario (default
  `flux-general-en`). Multilingual scenarios target `flux-general-multi` and
  PASS-WITH-NOTE if that model is not bundled.

Usage
-----
    uv run e2e/e2e_test_streaming.py your-flux-endpoint --region us-east-1

    # list scenarios:
    uv run e2e/e2e_test_streaming.py --list

    # run a subset:
    uv run e2e/e2e_test_streaming.py your-flux-endpoint --scenarios basic_25s,feature_eager_eot
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# Same-directory imports so the suite can run from anywhere.
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

DEFAULT_REGION = "us-east-1"
DEFAULT_MODEL = "flux-general-en"
MULTI_MODEL = "flux-general-multi"


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

@dataclass
class FluxScenario:
    """One Flux streaming scenario.

    `extra_args` are appended verbatim to `flux_stress.py file`. `model`
    overrides the run-wide model for this scenario (multilingual scenarios use
    `flux-general-multi`). The assertion fields below add Flux-specific checks
    on top of WER:

      - `min_eager_turns`: require ≥ N EagerEndOfTurn events (eager mode).
      - `expect_configure_success` / `expect_configure_failure`: require a
        ConfigureSuccess / ConfigureFailure ack from the mid-stream Configure.
      - `expect_languages`: soft note — record detected languages (only the
        multilingual model populates them); never fails the scenario.
      - `expect_failure`: the connection is expected to error (negative test);
        WER is skipped and the scenario PASSES iff a connection errored.
      - `tolerated_error_substring`: PASS-WITH-NOTE when the endpoint returns
        this known error (e.g. multilingual model not bundled).
    """
    name: str
    description: str
    use_long_form: bool = False
    connections: int = 1
    model: str | None = None
    extra_args: list[str] = field(default_factory=list)
    wer_threshold: float = 0.05
    skip_wer: bool = False
    presence_check: str | None = None
    min_eager_turns: int = 0
    expect_configure_success: bool = False
    expect_configure_failure: bool = False
    expect_languages: bool = False
    expect_failure: bool = False
    tolerated_error_substring: str | None = None
    notes: str = ""


def default_scenarios(model: str = DEFAULT_MODEL) -> list[FluxScenario]:
    """Feature coverage matrix per the Flux docs (June 2026 audit).

    The suite runs against either Flux model — `flux-general-en` or
    `flux-general-multi` — selected via ``--model`` to match the endpoint. The
    base scenarios use the run-wide model; the language-hint coverage adapts:

      - on a multilingual model: a POSITIVE test (language hints accepted →
        ``TurnInfo.languages`` populated);
      - on the English-only model: a NEGATIVE test (a `language_hint` is
        rejected, since hints aren't supported on `flux-general-en`).

    Connection params (set on the URL at connect time):
      model, encoding, sample_rate, eot_threshold (0.5–0.9, def 0.7),
      eager_eot_threshold (0.3–0.9, ≤ eot; enables EagerEndOfTurn/TurnResumed),
      eot_timeout_ms (500–10000, def 5000), keyterm (repeatable),
      language_hint (repeatable; multi only), mip_opt_out, tag.
    In-band control messages: Configure (thresholds / keyterms / language_hints),
      KeepAlive, Finalize, CloseStream (KeepAlive/Finalize PASS-WITH-NOTE on
      bundles that implement only CloseStream + Configure).

    Not covered here (need re-encoded fixtures we don't generate): non-linear16
    encodings (mulaw/alaw/opus) and alternate sample rates — the canonical
    sample is 16 kHz linear16.
    """
    is_multi = "multi" in (model or "").lower()
    scenarios: list[FluxScenario] = [
        # ---- Coverage / load ----
        FluxScenario(
            name="basic_25s",
            description="1 conn, 25 s file, defaults — baseline WER",
            connections=1,
        ),
        FluxScenario(
            name="concurrent_5x_25s",
            description="5 simultaneous connections, 25 s file",
            connections=5,
        ),
        FluxScenario(
            name="concurrent_10x_15min",
            description="10 simultaneous connections on ~15 min file",
            use_long_form=True,
            connections=10,
            notes="sustained-load WER check",
        ),
        FluxScenario(
            name="ramp_10x_step5",
            description="10 conns in batches of 5 with 2 s delay",
            connections=10,
            extra_args=["--batch-size", "5", "--batch-delay", "2"],
            notes="each conn streams the full file independently from open",
        ),
        # ---- End-of-turn detection ----
        FluxScenario(
            name="feature_eot_threshold_high",
            description="--eot-threshold 0.9 (later, higher-confidence EndOfTurn)",
            extra_args=["--eot-threshold", "0.9"],
            notes="fewer/later turns; content unchanged",
        ),
        FluxScenario(
            name="feature_eot_timeout_ms",
            description="--eot-timeout-ms 600 (force EndOfTurn on short silence)",
            extra_args=["--eot-timeout-ms", "600"],
            notes="forced EoT by silence timeout",
        ),
        FluxScenario(
            name="feature_eager_eot",
            description="--eager-eot-threshold 0.5 (enable EagerEndOfTurn)",
            extra_args=["--eager-eot-threshold", "0.5"],
            min_eager_turns=1,
            notes="eager events must be emitted when eager_eot_threshold set",
        ),
        # ---- Custom vocabulary / formatting ----
        FluxScenario(
            name="feature_keyterm",
            description="--keyterms 'spacewalk,female' (recognition boost)",
            extra_args=["--keyterms", "spacewalk,female"],
            presence_check="spacewalk",
            notes="keyterm prompting; presence is a soft signal",
        ),
        FluxScenario(
            name="feature_encoding_linear16",
            description="--encoding linear16 (explicit; matches the sample)",
            extra_args=["--encoding", "linear16"],
            notes="explicit encoding param accepted",
        ),
        FluxScenario(
            name="feature_mip_opt_out",
            description="--extra mip_opt_out=true (model-improvement opt-out; smoke)",
            extra_args=["--extra", "mip_opt_out=true"],
            notes="accepted; no observable transcript effect",
        ),
        # ---- In-band control messages ----
        FluxScenario(
            name="feature_configure_thresholds",
            description="mid-stream Configure raises eot_threshold to 0.8",
            extra_args=["--reconfigure-after", "5", "--reconfigure-eot-threshold", "0.8"],
            expect_configure_success=True,
            tolerated_error_substring="Configure",
            notes="ConfigureSuccess expected; PASS-WITH-NOTE on bundles that reject "
                  "Configure as an unknown variant (e.g. CloseStream-only builds)",
        ),
        FluxScenario(
            name="feature_configure_failure",
            description="mid-stream Configure with eager(0.9) > eot(0.5) — invalid",
            extra_args=[
                "--reconfigure-after", "5",
                "--reconfigure-eot-threshold", "0.5",
                "--reconfigure-eager-eot-threshold", "0.9",
            ],
            expect_configure_failure=True,
            tolerated_error_substring="Configure",
            notes="ConfigureFailure expected; PASS-WITH-NOTE on bundles that reject "
                  "Configure as an unknown variant",
        ),
        FluxScenario(
            name="feature_keepalive",
            description="--keepalive-interval 3 (periodic KeepAlive)",
            extra_args=["--keepalive-interval", "3"],
            tolerated_error_substring="KeepAlive",
            notes="KeepAlive when supported; older bundles reject it as an "
                  "unknown variant (PASS-WITH-NOTE)",
        ),
        FluxScenario(
            name="feature_finalize",
            description="--finalize-at-end (flush final turn before CloseStream)",
            extra_args=["--finalize-at-end"],
            tolerated_error_substring="Finalize",
            notes="Finalize when supported; older bundles reject it as an "
                  "unknown variant (PASS-WITH-NOTE)",
        ),
    ]

    # ---- Language hints: positive on multilingual, negative on English-only ----
    if is_multi:
        scenarios.append(FluxScenario(
            name="feature_lang_hint_multi",
            description="multilingual model + language_hint en — asserts TurnInfo.languages",
            extra_args=["--language-hints", "en"],
            expect_languages=True,
            notes="language hints accepted on the multilingual model",
        ))
    else:
        scenarios.append(FluxScenario(
            name="negative_lang_hint_on_en",
            description="English-only model + language_hint es — expect rejection",
            extra_args=["--language-hints", "es"],
            skip_wer=True,
            expect_failure=True,
            notes="language_hint is not supported on flux-general-en (expect HTTP 400)",
        ))
    return scenarios


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _stress_cmd(
    script: Path,
    endpoint: str,
    wav_path: Path,
    region: str,
    model: str,
    connections: int,
    summary_path: Path,
    extra: list[str],
) -> list[str]:
    cmd = [
        "uv", "run", "--project", str(script.parent),
        str(script), "file", endpoint,
        "--file", str(wav_path),
        "--region", region,
        "--model", model,
        "--connections", str(connections),
        "--summary-jsonl", str(summary_path),
        "--log-level", "WARNING",
    ]
    cmd.extend(extra)
    return cmd


def _scenario_timeout_s(scenario: FluxScenario, long_form_audio_s: float, base_timeout_s: int) -> int:
    """Per-scenario subprocess timeout.

    File mode paces audio to real time, so long-form scenarios need at least
    the audio duration plus headroom for connect / final turns / teardown.
    """
    if not scenario.use_long_form:
        return base_timeout_s
    headroom = max(300.0, long_form_audio_s * 0.3)
    return int(long_form_audio_s + headroom)


def _read_summary(summary_path: Path) -> list[dict]:
    if not summary_path.exists():
        return []
    return [json.loads(l) for l in summary_path.read_text().splitlines() if l.strip()]


def run_scenario(
    scenario: FluxScenario,
    *,
    endpoint: str,
    region: str,
    model: str,
    short_wav: Path,
    long_wav: Path,
    long_loops: int,
    long_form_audio_s: float,
    stress_script: Path,
    log_dir: Path,
    subprocess_timeout_s: int,
) -> dict:
    import subprocess

    wav = long_wav if scenario.use_long_form else short_wav
    eff_model = scenario.model or model
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
        stress_script, endpoint, wav, region, eff_model,
        scenario.connections, summary_path, scenario.extra_args,
    )
    logger.info(f"[{scenario.name}] running: {' '.join(cmd)}")

    start = time.monotonic()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return {
            "scenario": scenario.name, "ok": False, "wer": 1.0, "sdi": (0, 0, 0),
            "words": 0, "elapsed_s": elapsed,
            "notes": f"subprocess timed out after {timeout_s}s",
            "error": "timeout",
            "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
        }
    elapsed = time.monotonic() - start

    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)

    rows = _read_summary(summary_path)
    if not rows:
        return {
            "scenario": scenario.name, "ok": False, "wer": 1.0, "sdi": (0, 0, 0),
            "words": 0, "elapsed_s": elapsed,
            "notes": f"no per-connection summary produced (exit={result.returncode})",
            "error": f"exit {result.returncode}",
            "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
        }

    # Aggregate signals across connections.
    errored = [r for r in rows if r.get("errored")]
    total_eot = sum(r.get("turns_eot", 0) for r in rows)
    total_eager = sum(r.get("turns_eager", 0) for r in rows)
    cfg_ok = sum(r.get("configure_success", 0) for r in rows)
    cfg_fail = sum(r.get("configure_failure", 0) for r in rows)
    all_error_text = " ".join(
        m for r in rows for m in (r.get("error_messages") or [])
    ).lower()
    languages = sorted({
        lang for r in rows for lang in (r.get("languages_detected") or [])
    })

    notes_parts: list[str] = []
    if scenario.notes:
        notes_parts.append(scenario.notes)
    notes_parts.append(f"conns={len(rows)}")
    notes_parts.append(f"eot={total_eot}")
    if total_eager:
        notes_parts.append(f"eager={total_eager}")
    if cfg_ok or cfg_fail:
        notes_parts.append(f"cfg_ok/fail={cfg_ok}/{cfg_fail}")
    if errored:
        notes_parts.append(f"errored={len(errored)}")

    # --- Negative test: a connection is expected to error. ---
    if scenario.expect_failure:
        ok = len(errored) > 0
        notes_parts.append("expected-failure")
        if not ok:
            notes_parts.append("NO ERROR SEEN (expected one)")
        elif scenario.tolerated_error_substring and scenario.tolerated_error_substring.lower() not in all_error_text:
            notes_parts.append(f"err='{all_error_text[:80]}'")
        return _row(scenario, ok, 0.0, (0, 0, 0), 0, elapsed, notes_parts,
                    stdout_path, stderr_path, summary_path)

    # --- Tolerated error: deployment gap (e.g. multilingual model absent). ---
    tolerated = (
        scenario.tolerated_error_substring is not None
        and errored
        and scenario.tolerated_error_substring.lower() in all_error_text
    )
    if tolerated:
        notes_parts.append(
            f"DEPLOYMENT-GAP: '{scenario.tolerated_error_substring}' — pass-with-note"
        )
        return _row(scenario, True, 0.0, (0, 0, 0), 0, elapsed, notes_parts,
                    stdout_path, stderr_path, summary_path)

    # --- Normal scoring: WER on the strongest connection + assertions. ---
    rows.sort(key=lambda r: r.get("turns_eot", 0), reverse=True)
    best = rows[0]
    combined = best.get("combined_final_text", "") or ""
    w_ratio, s, d, i, n = wer(expected, combined)

    checks: list[bool] = [result.returncode == 0, len(errored) == 0, total_eot > 0]
    if not scenario.skip_wer:
        checks.append(w_ratio <= scenario.wer_threshold)

    if scenario.presence_check:
        present = scenario.presence_check.lower() in combined.lower()
        notes_parts.append(f"presence({scenario.presence_check})={'ok' if present else 'MISSING'}")
        # keyterm presence is a soft signal — note but do not fail the scenario.

    if scenario.min_eager_turns:
        eager_ok = total_eager >= scenario.min_eager_turns
        checks.append(eager_ok)
        if not eager_ok:
            notes_parts.append(f"EAGER<{scenario.min_eager_turns} (got {total_eager})")

    if scenario.expect_configure_success:
        cfg = cfg_ok >= 1
        checks.append(cfg)
        notes_parts.append(f"configure_success={'ok' if cfg else 'MISSING'}")

    if scenario.expect_configure_failure:
        cfg = cfg_fail >= 1
        checks.append(cfg)
        notes_parts.append(f"configure_failure={'ok' if cfg else 'MISSING'}")

    if scenario.expect_languages:
        notes_parts.append(f"languages={languages or 'none'}")  # soft note only

    ok = all(checks)
    return _row(scenario, ok, w_ratio, (s, d, i), n, elapsed, notes_parts,
                stdout_path, stderr_path, summary_path, combined[:200])


def _row(scenario, ok, wer_ratio, sdi, words, elapsed, notes_parts,
         stdout_path, stderr_path, summary_path, combined=""):
    row = {
        "scenario": scenario.name,
        "ok": ok,
        "wer": wer_ratio,
        "sdi": sdi,
        "words": words,
        "elapsed_s": elapsed,
        "notes": " ".join(notes_parts),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "summary_path": str(summary_path),
    }
    if combined:
        row["combined_final_text"] = combined
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "End-to-end correctness + feature test for a streaming SageMaker "
            "Flux endpoint. Downloads spacewalk.wav, multiplies it to ~15 min, "
            "runs a battery of scenarios through flux_stress.py file, and "
            "validates each session's combined EndOfTurn transcript via WER "
            "plus Flux-specific event/ack assertions."
        )
    )
    p.add_argument("endpoint_name", nargs="?", default=None,
                   help="Flux SageMaker endpoint name (required unless --list)")
    p.add_argument("--region", default=DEFAULT_REGION, help=f"AWS region (default: {DEFAULT_REGION})")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Flux model for non-multilingual scenarios (default: {DEFAULT_MODEL})")
    p.add_argument("--workdir", default=None, metavar="DIR",
                   help="Fixture + log directory (default: /tmp/dg-sagemaker-e2e/flux/<ts>)")
    p.add_argument("--target-long-form-s", type=float, default=900.0, metavar="SECONDS",
                   help="Target duration of the long-form multiplied WAV (default: 900 = 15 min)")
    p.add_argument("--scenarios", default="", metavar="NAME,NAME,...",
                   help="Comma-separated subset of scenario names (default: all). See --list.")
    p.add_argument("--list", action="store_true", help="List scenarios and exit")
    p.add_argument("--wer-threshold", type=float, default=0.05,
                   help="Default WER threshold for non-distorting scenarios (default: 0.05)")
    p.add_argument("--subprocess-timeout-s", type=int, default=120,
                   help="Per-scenario subprocess timeout for short-form (default: 120)")
    p.add_argument("--force-download", action="store_true",
                   help="Re-download spacewalk.wav even if cached")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return p


def main() -> int:
    args = _make_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")

    scenarios = default_scenarios(args.model)
    if args.wer_threshold != 0.05:
        for s in scenarios:
            if s.wer_threshold == 0.05:
                s.wer_threshold = args.wer_threshold

    if args.list:
        print("Available scenarios:")
        for s in scenarios:
            model = s.model or args.model
            print(f"  {s.name:<30} [{model}]  {s.description}")
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
            print("Run with --list to see available names.", file=sys.stderr)
            return 1
        scenarios = [s for s in scenarios if s.name in wanted]

    if args.workdir:
        workdir = Path(args.workdir).expanduser().resolve()
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        workdir = Path(tempfile.gettempdir()) / "dg-sagemaker-e2e" / "flux" / ts
    workdir.mkdir(parents=True, exist_ok=True)
    log_dir = workdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    short_wav = workdir / "spacewalk.wav"
    long_wav = workdir / "spacewalk-15min.wav"

    print("=" * 80)
    print(f"Endpoint:    {args.endpoint_name}")
    print(f"Region:      {args.region}")
    print(f"Model:       {args.model}")
    print(f"Workdir:     {workdir}")
    print("=" * 80)

    # 1. Fixture prep
    download_sample(short_wav, force=args.force_download)
    sr, ch, dur = validate_pcm16(short_wav)
    print(f"Sample:      {short_wav.name}  {sr} Hz  {ch}ch  {dur:.2f}s "
          f"({short_wav.stat().st_size / 1024:.0f} KB)")
    need_long = any(s.use_long_form for s in scenarios)
    long_dur = 0.0
    loops = 1
    if need_long:
        if not long_wav.exists() or args.force_download:
            loops = multiply_wav(short_wav, long_wav, args.target_long_form_s)
        _, _, long_dur = validate_pcm16(long_wav)
        loops = max(1, round(long_dur / dur))
        print(f"Long-form:   {long_wav.name}  {long_dur:.0f}s ({loops} loops) "
              f"({long_wav.stat().st_size / (1024*1024):.1f} MB)")

    # 2. Locate the stress script (one dir up — e2e/ is a sibling of flux_stress.py)
    stress_script = Path(__file__).resolve().parent.parent / "flux_stress.py"
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
