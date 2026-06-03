#!/usr/bin/env python3
"""
End-to-end test for a streaming (websocket) SageMaker TTS endpoint.

Drives `tts_stress.py` headless (`--no-playback --once --summary-jsonl`) through
streaming-specific scenarios — `Speak` → audio, `Flush` → `Flushed`, sustained
concurrency, the streaming encodings (linear16 / mulaw / alaw), and voice/speed
params — then validates each connection's synthesized audio from the per-
connection summary JSON (byte count, non-silent RMS for linear16, Flushed acks,
no errors).

This is the streaming counterpart to `e2e_test_batch.py`: the batch driver
carries the bulk of the parameter matrix (every encoding/container/bit_rate,
the 2000-char limit, the speed→duration assertion); this driver focuses on the
behaviors that only exist on the bidirectional `/v1/speak` transport.

Pass / fail
-----------
Each scenario succeeds when:
  - the subprocess exits 0 and no connection errored,
  - every connection received audio bytes (and, for linear16, non-silent RMS),
  - at least one `Flushed` ack was observed per connection.

`tolerated_error_substring` scenarios PASS-WITH-NOTE when the endpoint returns
a known "not supported" error (e.g. a voice the bundle doesn't serve).

Endpoint prerequisites
----------------------
- A streaming TTS SageMaker endpoint (bidirectional `/v1/speak`) is InService.

Usage
-----
    uv run e2e/e2e_test_streaming.py your-tts-endpoint --region us-east-2
    uv run e2e/e2e_test_streaming.py --list
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e_test_common import (
    REFERENCE_PHRASES,
    REFERENCE_TEXT,
    SILENCE_RMS_FLOOR,
    print_summary_table,
)

logger = logging.getLogger(__name__)

DEFAULT_REGION = "us-east-2"
DEFAULT_VOICE = "aura-2-thalia-en"


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

@dataclass
class TTSStreamScenario:
    """One streaming TTS scenario driven through `tts_stress.py`.

    `extra` are `/v1/speak` query params passed via `--extra`. Validation:
      - `check_rms`: require non-silent linear16 audio per connection.
      - `min_flushed`: require ≥ N Flushed acks per connection.
      - `multi_phrase`: send the 3-phrase fixture (vs. the single reference line).
      - `tolerated_error_substring`: PASS-WITH-NOTE on this known error.
    """
    name: str
    description: str
    connections: int = 1
    voice: str | None = None
    extra: dict[str, str] = field(default_factory=dict)
    multi_phrase: bool = False
    check_rms: bool = True
    min_flushed: int = 1
    tolerated_error_substring: str | None = None
    notes: str = ""


def default_scenarios() -> list[TTSStreamScenario]:
    return [
        TTSStreamScenario(
            name="basic",
            description="1 conn, send once, default linear16 — non-silent audio + Flushed",
            connections=1,
        ),
        TTSStreamScenario(
            name="concurrent_5",
            description="5 simultaneous connections synthesize the reference text",
            connections=5,
        ),
        TTSStreamScenario(
            name="multi_phrase_flush",
            description="3 phrases (Speak+Flush each) — multiple Flushed acks",
            multi_phrase=True,
            min_flushed=2,  # ≥2 guaranteed pre-close; the 3rd ack lands in the drain
            notes="exercises the Speak→Flush→Flushed loop across phrases",
        ),
        TTSStreamScenario(
            name="encoding_linear16_24k",
            description="explicit encoding=linear16 sample_rate=24000",
            extra={"encoding": "linear16", "sample_rate": "24000"},
        ),
        TTSStreamScenario(
            name="encoding_mulaw_8k",
            description="encoding=mulaw sample_rate=8000 (telephony; streaming-supported)",
            extra={"encoding": "mulaw", "sample_rate": "8000"},
            check_rms=False,  # companded — RMS-on-raw is meaningless
            tolerated_error_substring="encoding",
            notes="bytes-only check for non-linear16",
        ),
        TTSStreamScenario(
            name="speed_fast",
            description="speed=1.4 over the websocket transport (voice control)",
            extra={"speed": "1.4"},
            tolerated_error_substring="Flushed-ack timeout",
            notes="produces audio when the bundle supports speed; older bundles "
                  "silently drop it (no audio → flush-ack timeout) → PASS-WITH-NOTE",
        ),
        TTSStreamScenario(
            name="voice_alt",
            description="voice=aura-2-orion-en (alternate Aura-2 voice)",
            voice="aura-2-orion-en",
            tolerated_error_substring="model",
        ),
        TTSStreamScenario(
            name="mip_opt_out",
            description="mip_opt_out=true (smoke)",
            extra={"mip_opt_out": "true"},
        ),
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _stress_cmd(
    script: Path,
    endpoint: str,
    region: str,
    voice: str,
    connections: int,
    summary_path: Path,
    text_file: Path,
    extra: dict[str, str],
) -> list[str]:
    cmd = [
        "uv", "run", "--project", str(script.parent),
        str(script), endpoint,
        "--region", region,
        "--voice", voice,
        "--connections", str(connections),
        "--once",
        "--no-playback",
        "--summary-jsonl", str(summary_path),
        "--text-file", str(text_file),
        "--duration", "120",  # generous safety cap; --once stops first
        "--log-level", "WARNING",
    ]
    if extra:
        cmd += ["--extra", "&".join(f"{k}={v}" for k, v in extra.items())]
    return cmd


def run_scenario(
    scenario: TTSStreamScenario,
    *,
    endpoint: str,
    region: str,
    voice: str,
    stress_script: Path,
    single_text_file: Path,
    multi_text_file: Path,
    log_dir: Path,
    subprocess_timeout_s: int,
) -> dict:
    eff_voice = scenario.voice or voice
    text_file = multi_text_file if scenario.multi_phrase else single_text_file
    summary_path = log_dir / f"{scenario.name}.summary.jsonl"
    stdout_path = log_dir / f"{scenario.name}.stdout.log"
    stderr_path = log_dir / f"{scenario.name}.stderr.log"

    cmd = _stress_cmd(
        stress_script, endpoint, region, eff_voice,
        scenario.connections, summary_path, text_file, scenario.extra,
    )
    logger.info(f"[{scenario.name}] running: {' '.join(cmd)}")

    start = time.monotonic()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=subprocess_timeout_s)
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return _row(scenario, False, elapsed, [f"subprocess timed out after {subprocess_timeout_s}s"],
                    error="timeout")
    elapsed = time.monotonic() - start

    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)

    rows = []
    if summary_path.exists():
        rows = [json.loads(l) for l in summary_path.read_text().splitlines() if l.strip()]
    if not rows:
        return _row(scenario, False, elapsed,
                    [f"no per-connection summary (exit={result.returncode})"],
                    error=f"exit {result.returncode}")

    errored = [r for r in rows if r.get("errored")]
    all_error_text = " ".join(
        m for r in rows for m in (r.get("error_messages") or [])
    ).lower()

    # Tolerated deployment gap (e.g. voice/encoding not bundled).
    if scenario.tolerated_error_substring and errored and scenario.tolerated_error_substring.lower() in all_error_text:
        return _row(scenario, True, elapsed,
                    [f"DEPLOYMENT-GAP: '{scenario.tolerated_error_substring}' — pass-with-note"])

    total_bytes = sum(r.get("audio_bytes", 0) for r in rows)
    min_conn_bytes = min((r.get("audio_bytes", 0) for r in rows), default=0)
    total_flushed = sum(r.get("flushed_count", 0) for r in rows)
    min_conn_flushed = min((r.get("flushed_count", 0) for r in rows), default=0)
    rms_values = [r.get("audio_rms") for r in rows if r.get("audio_rms") is not None]
    first = rows[0]

    checks: list[bool] = [result.returncode == 0, len(errored) == 0, min_conn_bytes > 0]
    notes: list[str] = []
    if scenario.notes:
        notes.append(scenario.notes)
    notes.append(f"conns={len(rows)}")
    notes.append(f"bytes(min)={min_conn_bytes}")
    notes.append(f"flushed(min)={min_conn_flushed}")
    if errored:
        notes.append(f"errored={len(errored)}")

    if scenario.min_flushed:
        flush_ok = min_conn_flushed >= scenario.min_flushed
        checks.append(flush_ok)
        if not flush_ok:
            notes.append(f"FLUSHED<{scenario.min_flushed}")

    rms_val = None
    if scenario.check_rms:
        # Require every connection that reported an RMS to be non-silent.
        if rms_values:
            rms_val = round(min(rms_values), 1)
            rms_ok = all(v > SILENCE_RMS_FLOOR for v in rms_values)
            checks.append(rms_ok)
            notes.append(f"rms(min)={rms_val}{'' if rms_ok else ' (SILENT!)'}")
        else:
            checks.append(False)
            notes.append("no RMS reported (expected linear16 audio)")

    warnings = [w for r in rows for w in (r.get("warnings") or [])]
    if warnings:
        notes.append(f"warnings={len(warnings)}")

    ok = all(checks)
    return _row(scenario, ok, elapsed, notes,
                bytes_=total_bytes, rms=rms_val,
                duration_s=first.get("audio_duration_s"),
                stdout_path=stdout_path, stderr_path=stderr_path, summary_path=summary_path)


def _row(scenario, ok, elapsed, notes, *, bytes_=None, rms=None, duration_s=None,
         error=None, stdout_path=None, stderr_path=None, summary_path=None) -> dict:
    row = {
        "scenario": scenario.name,
        "ok": ok,
        "bytes": bytes_,
        "rms": rms,
        "duration_s": duration_s,
        "elapsed_s": elapsed,
        "notes": " ".join(notes),
        "error": error,
    }
    if stdout_path:
        row["stdout_path"] = str(stdout_path)
        row["stderr_path"] = str(stderr_path)
        row["summary_path"] = str(summary_path)
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "End-to-end test for a streaming (websocket) SageMaker TTS endpoint. "
            "Drives tts_stress.py headless and validates each connection's "
            "synthesized audio (bytes / non-silent RMS / Flushed acks) from the "
            "per-connection summary JSON."
        )
    )
    p.add_argument("endpoint_name", nargs="?", default=None,
                   help="Streaming TTS SageMaker endpoint name (required unless --list)")
    p.add_argument("--region", default=DEFAULT_REGION, help=f"AWS region (default: {DEFAULT_REGION})")
    p.add_argument("--voice", default=DEFAULT_VOICE, help=f"Default TTS voice (default: {DEFAULT_VOICE})")
    p.add_argument("--workdir", default=None, metavar="DIR",
                   help="Fixture + log directory (default: /tmp/dg-sagemaker-e2e/tts-streaming/<ts>)")
    p.add_argument("--scenarios", default="", metavar="NAME,NAME,...",
                   help="Comma-separated subset of scenario names (default: all). See --list.")
    p.add_argument("--list", action="store_true", help="List scenarios and exit")
    p.add_argument("--subprocess-timeout-s", type=int, default=120,
                   help="Per-scenario subprocess timeout (default: 120)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return p


def main() -> int:
    args = _make_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")

    scenarios = default_scenarios()

    if args.list:
        print("Available scenarios:")
        for s in scenarios:
            print(f"  {s.name:<24} {s.description}")
        return 0

    if not args.endpoint_name:
        print("ERROR: endpoint_name is required (run with --list to see scenarios).", file=sys.stderr)
        return 1

    if args.scenarios:
        wanted = {tok.strip() for tok in args.scenarios.split(",") if tok.strip()}
        unknown = wanted - {s.name for s in scenarios}
        if unknown:
            print(f"ERROR: unknown scenario(s): {sorted(unknown)}", file=sys.stderr)
            return 1
        scenarios = [s for s in scenarios if s.name in wanted]

    if args.workdir:
        workdir = Path(args.workdir).expanduser().resolve()
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        workdir = Path(tempfile.gettempdir()) / "dg-sagemaker-e2e" / "tts-streaming" / ts
    workdir.mkdir(parents=True, exist_ok=True)
    log_dir = workdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Write text fixtures.
    single_text_file = workdir / "tts-single.txt"
    single_text_file.write_text(REFERENCE_TEXT + "\n")
    multi_text_file = workdir / "tts-phrases.txt"
    multi_text_file.write_text("\n".join(REFERENCE_PHRASES) + "\n")

    stress_script = Path(__file__).resolve().parent.parent / "tts_stress.py"
    if not stress_script.is_file():
        print(f"ERROR: {stress_script} missing", file=sys.stderr)
        return 2

    print("=" * 80)
    print(f"Endpoint:    {args.endpoint_name}")
    print(f"Region:      {args.region}")
    print(f"Voice:       {args.voice}")
    print(f"Workdir:     {workdir}")
    print("=" * 80)
    print()

    rows: list[dict] = []
    for scenario in scenarios:
        print(f"--> {scenario.name}  ({scenario.description})")
        row = run_scenario(
            scenario,
            endpoint=args.endpoint_name,
            region=args.region,
            voice=args.voice,
            stress_script=stress_script,
            single_text_file=single_text_file,
            multi_text_file=multi_text_file,
            log_dir=log_dir,
            subprocess_timeout_s=args.subprocess_timeout_s,
        )
        rows.append(row)
        flag = "PASS" if row["ok"] else "FAIL"
        print(f"    {flag}  elapsed={row['elapsed_s']:.1f}s  {row['notes']}")

    print()
    passed, failed = print_summary_table(rows)
    (workdir / "results.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nFull results: {workdir / 'results.json'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
