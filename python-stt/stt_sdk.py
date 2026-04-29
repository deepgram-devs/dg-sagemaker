#!/usr/bin/env python3
"""
Smoke test a Deepgram Nova STT SageMaker endpoint through the Deepgram Python SDK.

This script uses the official Deepgram SDK with the deepgram-sagemaker transport,
so SageMaker HTTP/2 bidirectional streaming is hidden behind the normal
client.listen.v1 WebSocket interface.
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
from deepgram.listen.v1.types import (
    ListenV1CloseStream,
    ListenV1Metadata,
    ListenV1Results,
    ListenV1SpeechStarted,
    ListenV1UtteranceEnd,
)
from deepgram_sagemaker import SageMakerTransportFactory


DEFAULT_REGION = "us-east-2"
DEFAULT_MODEL = "nova-3"
DEFAULT_CHUNK_MS = 80

ListenV1Message = Union[
    ListenV1Results,
    ListenV1Metadata,
    ListenV1UtteranceEnd,
    ListenV1SpeechStarted,
]


def parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def api_bool(value: bool) -> str:
    return "true" if value else "false"


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
        if sample_width != 2:
            raise ValueError(
                f"{args.file} is {sample_width * 8}-bit audio. "
                "Deepgram streaming STT expects 16-bit PCM (linear16). "
                "Convert it with: ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 output.wav"
            )

        chunk_size = wav_chunk_size(sample_rate, channels, sample_width, args.chunk_ms)
        chunk_delay = args.chunk_ms / 1000

        async with client.listen.v1.connect(
            model=args.model,
            language=args.language,
            encoding="linear16",
            sample_rate=sample_rate,
            channels=channels if channels > 1 else None,
            punctuate=api_bool(args.punctuate),
            interim_results=api_bool(args.interim_results),
            diarize=api_bool(args.diarize),
            keyterm=parse_csv(args.keyterms),
            keywords=parse_csv(args.keywords),
        ) as connection:
            def on_message(message: ListenV1Message) -> None:
                if isinstance(message, ListenV1Results):
                    alternatives = message.channel.alternatives if message.channel else []
                    if alternatives and alternatives[0].transcript:
                        label = "final" if message.is_final else "interim"
                        print(f"[{label}] {alternatives[0].transcript}", flush=True)
                elif isinstance(message, ListenV1SpeechStarted):
                    print("[event] speech started", flush=True)
                elif isinstance(message, ListenV1UtteranceEnd):
                    print("[event] utterance end", flush=True)

            connection.on(EventType.OPEN, lambda _: print("Connection opened", flush=True))
            connection.on(EventType.MESSAGE, on_message)
            connection.on(EventType.CLOSE, lambda _: print("Connection closed", flush=True))
            connection.on(EventType.ERROR, lambda error: print(f"Connection error: {error}", file=sys.stderr, flush=True))

            listen_task = asyncio.create_task(connection.start_listening())
            print(
                f"Streaming {args.file} to {args.endpoint_name} "
                f"({sample_rate} Hz, {channels} channel(s), {args.chunk_ms} ms chunks)",
                flush=True,
            )

            while True:
                chunk = wav_file.readframes(chunk_size // (channels * sample_width))
                if not chunk:
                    break
                await connection.send_media(chunk)
                await asyncio.sleep(chunk_delay)

            await connection.send_close_stream(ListenV1CloseStream(type="CloseStream"))
            await asyncio.sleep(args.drain_seconds)
            listen_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await listen_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream a PCM WAV file to a Deepgram STT SageMaker endpoint via the Deepgram Python SDK."
    )
    parser.add_argument("endpoint_name", help="SageMaker endpoint name")
    parser.add_argument("--file", required=True, type=Path, help="16-bit PCM WAV file to stream")
    parser.add_argument("--region", default=DEFAULT_REGION, help=f"AWS region (default: {DEFAULT_REGION})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Deepgram STT model (default: {DEFAULT_MODEL})")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    parser.add_argument("--punctuate", type=parse_bool, default=True, help="Enable punctuation (default: true)")
    parser.add_argument("--interim-results", type=parse_bool, default=True, help="Emit interim results (default: true)")
    parser.add_argument("--diarize", type=parse_bool, default=False, help="Enable speaker diarization (default: false)")
    parser.add_argument("--keywords", help="Comma-separated nova-2 keyword boosts, e.g. Deepgram:5,SageMaker:10")
    parser.add_argument("--keyterms", help="Comma-separated nova-3 keyterms, e.g. Deepgram,SageMaker")
    parser.add_argument("--chunk-ms", type=int, default=DEFAULT_CHUNK_MS, help="Audio chunk duration in ms (default: 80)")
    parser.add_argument("--drain-seconds", type=float, default=2.0, help="Seconds to wait for final responses (default: 2)")
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    try:
        await stream_wav(args)
    except Exception as exc:
        print(
            "STT SDK smoke test failed. Check that the endpoint name and region are correct, "
            "AWS credentials allow sagemaker:InvokeEndpointWithBidirectionalStream, "
            "the endpoint is InService, and the WAV file is 16-bit PCM. "
            f"Original error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    asyncio.run(main())
