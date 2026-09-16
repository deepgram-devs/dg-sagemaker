#!/usr/bin/env python3
"""
Directed cross-stream contamination test against a REAL SageMaker Flux endpoint.

Same methodology as the customer-investigations raw-websocket script
(`cross_stream_poison_test.py`, used against Jack Kearney's non-SageMaker
coder.deepgram.com dev workspace), ported to the SageMaker bidi-stream
transport (`aws_sdk_sagemaker_runtime_http2`) so it can validate the actual
deployed image/model:

  1. Open connection A, loop the test utterance --poison-loops times (so its
     internal state should be in the "elevated" regime), then close it.
  2. Immediately (no jitter) open a brand new connection B on the SAME shared
     SageMaker client and stream ONE fresh turn. If B's first-ever EndOfTurn
     confidence lands in the elevated band, that's evidence B inherited A's
     leftover state.

Reuses DeepgramFluxConnection from flux_stress.py (same directory) rather
than reimplementing the SageMaker transport.
"""
import argparse
import asyncio
import itertools
import json
import time
import wave

from flux_stress import (
    DeepgramFluxConnection,
    DEFAULT_REGION,
    DEFAULT_MODEL,
)
from aws_sdk_sagemaker_runtime_http2.client import SageMakerRuntimeHTTP2Client
from aws_sdk_sagemaker_runtime_http2.config import Config, HTTPAuthSchemeResolver
from smithy_aws_core.identity import EnvironmentCredentialsResolver
from smithy_aws_core.auth.sigv4 import SigV4AuthScheme
import boto3
import os

AUDIO_CHUNK_MS = 80
BASELINE_HIGH = 0.65  # isolated turn-1 baseline is ~0.55; clearly above this = "elevated"

_conn_id_counter = itertools.count(1)


def load_wav(path):
    wf = wave.open(str(path), "rb")
    sample_rate = wf.getframerate()
    sample_width = wf.getsampwidth()
    raw = wf.readframes(wf.getnframes())
    wf.close()
    return raw, sample_rate, sample_width


def build_client(region: str) -> SageMakerRuntimeHTTP2Client:
    session = boto3.Session(region_name=region)
    credentials = session.get_credentials()
    frozen = credentials.get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
    if frozen.token:
        os.environ["AWS_SESSION_TOKEN"] = frozen.token

    config = Config(
        endpoint_uri=f"https://runtime.sagemaker.{region}.amazonaws.com:8443",
        region=region,
        aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        auth_scheme_resolver=HTTPAuthSchemeResolver(),
        auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="sagemaker")},
    )
    return SageMakerRuntimeHTTP2Client(config=config)


class CapturingFluxConnection(DeepgramFluxConnection):
    """Adds a per-stream_one_turn EndOfTurn wait + raw (event, confidence) capture."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.captured: list[tuple[int, str, float | None]] = []
        self._turn_done = asyncio.Event()

    def _handle_turn_info(self, msg: dict):
        event = msg.get("event", "")
        eot_confidence = msg.get("end_of_turn_confidence")
        turn_index = msg.get("turn_index", self.turn_index)
        if event in ("EagerEndOfTurn", "EndOfTurn"):
            self.captured.append((turn_index, event, eot_confidence))
        super()._handle_turn_info(msg)
        if event == "EndOfTurn":
            self._turn_done.set()

    async def stream_one_turn(self, raw: bytes, sample_rate: int, sample_width: int,
                               chunk_ms: int = AUDIO_CHUNK_MS):
        """Streams the clip once on an already-open session; returns (eager_conf, eot_conf)."""
        self._turn_done.clear()
        before = len(self.captured)
        frame_bytes = int(sample_rate * chunk_ms / 1000) * sample_width
        silence_frame = b"\x00" * frame_bytes
        total_frames = (len(raw) + frame_bytes - 1) // frame_bytes

        t0 = time.perf_counter()
        for i in range(total_frames):
            off = i * frame_bytes
            chunk = raw[off:off + frame_bytes]
            if len(chunk) < frame_bytes:
                chunk = chunk + silence_frame[: frame_bytes - len(chunk)]
            target = t0 + (i * chunk_ms / 1000.0)
            now = time.perf_counter()
            if target > now:
                await asyncio.sleep(target - now)
            await self.send_audio_chunk(chunk)

        for _ in range(30):
            if self._turn_done.is_set():
                break
            await asyncio.sleep(chunk_ms / 1000.0)
            await self.send_audio_chunk(silence_frame)

        if not self._turn_done.is_set():
            try:
                await asyncio.wait_for(self._turn_done.wait(), timeout=3)
            except asyncio.TimeoutError:
                pass

        new = self.captured[before:]
        eager_conf = next((c for _, e, c in new if e == "EagerEndOfTurn"), None)
        eot_conf = next((c for _, e, c in new if e == "EndOfTurn"), None)
        return eager_conf, eot_conf


async def poison_then_probe(client, endpoint_name, model, raw, sample_rate, sample_width,
                             poison_loops, pair_id, results, eot_threshold, eager_eot_threshold,
                             eot_timeout_ms):
    conn_a = CapturingFluxConnection(next(_conn_id_counter), client, endpoint_name)
    poison_confs = []
    try:
        await conn_a.start_session(
            sample_rate=sample_rate, model=model,
            eot_threshold=eot_threshold, eager_eot_threshold=eager_eot_threshold,
            eot_timeout_ms=eot_timeout_ms,
        )
        for _ in range(poison_loops):
            _, eot_conf = await conn_a.stream_one_turn(raw, sample_rate, sample_width)
            poison_confs.append(eot_conf)
        await conn_a.end_session()
    except Exception as e:
        results.append({"pair_id": pair_id, "outcome": f"poison_errored:{e!r}"})
        return

    # No jitter -- open the probe connection as fast as possible after A closes.
    conn_b = CapturingFluxConnection(next(_conn_id_counter), client, endpoint_name)
    try:
        await conn_b.start_session(
            sample_rate=sample_rate, model=model,
            eot_threshold=eot_threshold, eager_eot_threshold=eager_eot_threshold,
            eot_timeout_ms=eot_timeout_ms,
        )
        eager_conf, eot_conf = await conn_b.stream_one_turn(raw, sample_rate, sample_width)
        await conn_b.end_session()
    except Exception as e:
        results.append({"pair_id": pair_id, "outcome": f"probe_errored:{e!r}", "poison_confs": poison_confs})
        return

    elevated = eot_conf is not None and eot_conf > BASELINE_HIGH
    results.append({
        "pair_id": pair_id,
        "outcome": "ok",
        "conn_a_id": conn_a.connection_id,
        "conn_b_id": conn_b.connection_id,
        "poison_confs": poison_confs,
        "probe_eager_conf": eager_conf,
        "probe_eot_conf": eot_conf,
        "probe_elevated": elevated,
    })


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint-name", required=True)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--file", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--eot-threshold", type=float, default=0.7)
    p.add_argument("--eager-eot-threshold", type=float, default=0.5)
    p.add_argument("--eot-timeout-ms", type=int, default=500)
    p.add_argument("--poison-loops", type=int, default=3)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--concurrent-pairs", type=int, default=1,
                    help="how many (poison,probe) pairs to run simultaneously per trial batch")
    p.add_argument("--results-jsonl", default=None)
    args = p.parse_args()

    raw, sample_rate, sample_width = load_wav(args.file)
    client = build_client(args.region)

    results = []
    batches = (args.trials + args.concurrent_pairs - 1) // args.concurrent_pairs
    pair_counter = 0
    for b in range(batches):
        n = min(args.concurrent_pairs, args.trials - pair_counter)
        tasks = []
        for _ in range(n):
            tasks.append(poison_then_probe(
                client, args.endpoint_name, args.model, raw, sample_rate, sample_width,
                args.poison_loops, pair_counter, results,
                args.eot_threshold, args.eager_eot_threshold, args.eot_timeout_ms,
            ))
            pair_counter += 1
        await asyncio.gather(*tasks, return_exceptions=True)
        print(f"batch {b+1}/{batches} done ({pair_counter}/{args.trials} pairs)")

    if args.results_jsonl:
        with open(args.results_jsonl, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    print()
    print("=" * 60)
    print(f"  SUMMARY: {args.trials} poison({args.poison_loops})->probe(1) pairs, "
          f"{args.concurrent_pairs} concurrent per batch, endpoint={args.endpoint_name}")
    print("=" * 60)
    ok = [r for r in results if r["outcome"] == "ok"]
    errored = [r for r in results if r["outcome"] != "ok"]
    print(f"  ok={len(ok)} errored={len(errored)}")
    for r in errored:
        print(f"    pair {r['pair_id']}: {r['outcome']}")
    elevated = [r for r in ok if r["probe_elevated"]]
    print(f"  probe connections landing ABOVE {BASELINE_HIGH} (elevated/contaminated): "
          f"{len(elevated)}/{len(ok)}")
    for r in ok:
        flag = "ELEVATED <-- contamination?" if r["probe_elevated"] else "baseline"
        print(f"    pair {r['pair_id']}: poison_confs={r['poison_confs']}  "
              f"probe_eager={r['probe_eager_conf']}  probe_eot={r['probe_eot_conf']}  {flag}")


if __name__ == "__main__":
    asyncio.run(main())
