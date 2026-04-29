#!/usr/bin/env python3
"""
Smoke test a Deepgram Flux SageMaker endpoint through the Deepgram Python SDK.

Flux uses the SDK's listen.v2 interface and the deepgram-sagemaker transport to
invoke SageMaker over HTTP/2 bidirectional streaming.
"""

import argparse
import asyncio
import contextlib
import sys
import wave
from pathlib import Path
from typing import Union

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.listen.v2.types import (
    ListenV2CloseStream,
    ListenV2ConfigureFailure,
    ListenV2Connected,
    ListenV2FatalError,
    ListenV2TurnInfo,
)
from deepgram_sagemaker import SageMakerTransportFactory


DEFAULT_REGION = "us-east-1"
DEFAULT_MODEL = "flux-general-en"
DEFAULT_CHUNK_MS = 80

ListenV2Message = Union[
    ListenV2Connected,
    ListenV2TurnInfo,
    ListenV2ConfigureFailure,
    ListenV2FatalError,
]


def parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def wav_chunk_size(sample_rate: int, channels: int, sample_width: int, chunk_ms: int) -> int:
    frames_per_chunk = max(1, int(sample_rate * chunk_ms / 1000))
    return frames_per_chunk * channels * sample_width


async def stream_wav(args: argparse.Namespace) -> None:
    factory = SageMakerTransportFactory(endpoint_name=args.endpoint_name, region=args.region)
    client = AsyncDeepgramClient(api_key="unused", transport_factory=factory)

    with wave.open(str(args.file), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        if sample_width != 2 or channels != 1:
            raise ValueError(
                f"{args.file} is {sample_width * 8}-bit audio with {channels} channel(s). "
                "Flux smoke tests expect mono 16-bit PCM. Convert with: "
                "ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 output.wav"
            )

        chunk_size = wav_chunk_size(sample_rate, channels, sample_width, args.chunk_ms)
        chunk_delay = args.chunk_ms / 1000

        async with client.listen.v2.connect(
            model=args.model,
            encoding="linear16",
            sample_rate=sample_rate,
            eot_threshold=args.eot_threshold,
            eager_eot_threshold=args.eager_eot_threshold,
            eot_timeout_ms=args.eot_timeout_ms,
            keyterm=parse_csv(args.keyterms),
        ) as connection:
            def print_turn(event: str, turn_index: int, transcript: str | None) -> None:
                transcript = transcript.strip() if transcript else ""
                if transcript:
                    print(f"[{event} turn={turn_index}] {transcript}", flush=True)
                else:
                    print(f"[{event} turn={turn_index}]", flush=True)

            def on_message(message: ListenV2Message) -> None:
                if isinstance(message, ListenV2Connected):
                    print("[event] connected", flush=True)
                elif isinstance(message, ListenV2TurnInfo):
                    print_turn(message.event, message.turn_index, message.transcript)
                elif isinstance(message, ListenV2ConfigureFailure):
                    print(f"[event] configure failed: {message}", file=sys.stderr, flush=True)
                elif isinstance(message, ListenV2FatalError):
                    print(f"[event] fatal error: {message}", file=sys.stderr, flush=True)
                elif isinstance(message, dict):
                    msg_type = message.get("type", "Unknown")
                    if msg_type == "Connected":
                        print("[event] connected", flush=True)
                    elif msg_type == "TurnInfo":
                        print_turn(
                            str(message.get("event", "Update")),
                            int(message.get("turn_index", 0)),
                            message.get("transcript"),
                        )
                    elif msg_type in {"ConfigureFailure", "Error"}:
                        print(f"[event] {msg_type}: {message}", file=sys.stderr, flush=True)
                    else:
                        print(f"[event] {msg_type}: {message}", flush=True)
                else:
                    print(f"[event] {getattr(message, 'type', type(message).__name__)}", flush=True)

            connection.on(EventType.OPEN, lambda _: print("Connection opened", flush=True))
            connection.on(EventType.MESSAGE, on_message)
            connection.on(EventType.CLOSE, lambda _: print("Connection closed", flush=True))
            connection.on(EventType.ERROR, lambda error: print(f"Connection error: {error}", file=sys.stderr, flush=True))

            listen_task = asyncio.create_task(connection.start_listening())
            print(
                f"Streaming {args.file} to {args.endpoint_name} "
                f"({sample_rate} Hz, {args.chunk_ms} ms Flux chunks)",
                flush=True,
            )

            while True:
                chunk = wav_file.readframes(chunk_size // (channels * sample_width))
                if not chunk:
                    break
                await connection.send_media(chunk)
                await asyncio.sleep(chunk_delay)

            await connection.send_close_stream(ListenV2CloseStream(type="CloseStream"))
            await asyncio.sleep(args.drain_seconds)
            listen_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await listen_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream a mono PCM WAV file to a Deepgram Flux SageMaker endpoint via the Deepgram Python SDK."
    )
    parser.add_argument("endpoint_name", help="SageMaker endpoint name")
    parser.add_argument("--file", required=True, type=Path, help="Mono 16-bit PCM WAV file to stream")
    parser.add_argument("--region", default=DEFAULT_REGION, help=f"AWS region (default: {DEFAULT_REGION})")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=["flux-general-en", "flux-general-multi"], help=f"Flux model (default: {DEFAULT_MODEL})")
    parser.add_argument("--eot-threshold", type=float, help="EndOfTurn confidence threshold, 0.5-0.9")
    parser.add_argument("--eager-eot-threshold", type=float, help="EagerEndOfTurn confidence threshold, 0.3-0.9")
    parser.add_argument("--eot-timeout-ms", type=int, help="Max silence before forced EndOfTurn, 500-10000 ms")
    parser.add_argument("--keyterms", help="Comma-separated keyterms, e.g. Deepgram,SageMaker")
    parser.add_argument("--chunk-ms", type=int, default=DEFAULT_CHUNK_MS, help="Audio chunk duration in ms (default: 80)")
    parser.add_argument("--drain-seconds", type=float, default=2.0, help="Seconds to wait for final responses (default: 2)")
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    try:
        await stream_wav(args)
    except Exception as exc:
        print(
            "Flux SDK smoke test failed. Check that the endpoint name and region are correct, "
            "AWS credentials allow sagemaker:InvokeEndpointWithBidirectionalStream, "
            "the endpoint is InService, the endpoint has listen_v2 enabled, and the WAV file "
            "is mono 16-bit PCM. "
            f"Original error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    asyncio.run(main())
