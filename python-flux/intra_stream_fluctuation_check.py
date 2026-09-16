#!/usr/bin/env python3
"""
Single-connection, multi-turn fluctuation check.

Confirms that within ONE still-open connection, looping the same audio
clip across several turns produces a genuinely EVOLVING (not frozen/
constant, not flattened-to-baseline) EndOfTurn confidence sequence -- the
expected signature of the intentional soft-reset state retention (turn
boundary = hard=False, preserves the EoT buffer). If the sequence were
flat (every turn == turn-0's baseline), that would suggest the fix
accidentally made soft resets behave like hard resets.
"""
import argparse
import asyncio
import wave

from cross_stream_poison_sagemaker import CapturingFluxConnection, build_client


def load_wav(path):
    wf = wave.open(str(path), "rb")
    sample_rate = wf.getframerate()
    sample_width = wf.getsampwidth()
    raw = wf.readframes(wf.getnframes())
    wf.close()
    return raw, sample_rate, sample_width


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint-name", required=True)
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--file", required=True)
    p.add_argument("--model", default="flux-general-multi")
    p.add_argument("--loops", type=int, default=8)
    args = p.parse_args()

    raw, sample_rate, sample_width = load_wav(args.file)
    client = build_client(args.region)

    conn = CapturingFluxConnection(1, client, args.endpoint_name)
    await conn.start_session(
        sample_rate=sample_rate, model=args.model,
        eot_threshold=0.7, eager_eot_threshold=0.5, eot_timeout_ms=500,
    )
    sequence = []
    for i in range(args.loops):
        eager_conf, eot_conf = await conn.stream_one_turn(raw, sample_rate, sample_width)
        sequence.append((i, eager_conf, eot_conf))
    await conn.end_session()

    print()
    print("=" * 60)
    print(f"  Single-connection {args.loops}-turn fluctuation check ({args.file.split('/')[-1]})")
    print("=" * 60)
    for i, eager, eot in sequence:
        print(f"  turn {i}: eager={eager}  eot={eot}")

    eot_values = [eot for _, _, eot in sequence if eot is not None]
    distinct = set(eot_values)
    print()
    print(f"  distinct EoT values across {len(eot_values)} turns: {len(distinct)}  -> {sorted(distinct)}")
    if len(distinct) <= 1:
        print("  ** FLAT ** -- no turn-to-turn fluctuation observed (unexpected)")
    else:
        print("  ** FLUCTUATING ** -- turn-to-turn state is evolving as intended")


if __name__ == "__main__":
    asyncio.run(main())
