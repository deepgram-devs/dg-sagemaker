#!/usr/bin/env python3
"""
Smoke test a Deepgram TTS SageMaker endpoint through the Deepgram Python SDK.

The script streams one text prompt through speak.v1 and writes the returned
audio bytes to a local file.
"""

import argparse
import asyncio
import contextlib
import json
import sys
from pathlib import Path

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.speak.v1.types import SpeakV1Close, SpeakV1Text
from deepgram_sagemaker import SageMakerTransportFactory


DEFAULT_REGION = "us-east-2"
DEFAULT_MODEL = "aura-2-thalia-en"
DEFAULT_TEXT = "Hello from Deepgram text to speech on Amazon SageMaker."
STREAMING_ENCODINGS = ("linear16", "mulaw", "alaw")
JSON_PREFIXES = (b"{", b"[")


def read_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return args.text_file.read_text(encoding="utf-8").strip()
    return args.text.strip()


def parse_json_payload(payload: bytes) -> dict | list | None:
    stripped = payload.lstrip()
    if not stripped.startswith(JSON_PREFIXES):
        return None
    try:
        return json.loads(stripped.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def describe_control_event(event: object) -> str:
    if isinstance(event, dict):
        msg_type = event.get("type", "Unknown")
        return f"{msg_type}: {event}"
    if isinstance(event, list):
        return f"JSON array: {event}"
    return repr(event)


async def synthesize(args: argparse.Namespace) -> None:
    text = read_text(args)
    if not text:
        raise ValueError("No text was provided. Pass text as an argument or use --text-file.")

    factory = SageMakerTransportFactory(endpoint_name=args.endpoint_name, region=args.region)
    client = AsyncDeepgramClient(api_key="unused", transport_factory=factory)
    closed = asyncio.Event()
    bytes_written = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        async with client.speak.v1.connect(
            model=args.model,
            encoding=args.encoding,
            sample_rate=args.sample_rate,
        ) as connection:
            def on_message(message) -> None:
                nonlocal bytes_written
                if isinstance(message, bytes):
                    control_event = parse_json_payload(message)
                    if control_event is not None:
                        print(
                            f"[event] {describe_control_event(control_event)}",
                            flush=True,
                        )
                        return

                    output.write(message)
                    bytes_written += len(message)
                    return

                if isinstance(message, dict):
                    msg_type = message.get("type", "Unknown")
                    print(f"[event] {msg_type}: {message}", flush=True)
                    return

                msg_type = getattr(message, "type", type(message).__name__)
                if msg_type == "Metadata":
                    request_id = getattr(message, "request_id", "unknown")
                    model_name = getattr(message, "model_name", "unknown")
                    print(f"[event] Metadata request_id={request_id} model={model_name}", flush=True)
                elif msg_type == "Warning":
                    code = getattr(message, "code", "UNKNOWN")
                    description = getattr(message, "description", "")
                    print(f"[event] Warning {code}: {description}", file=sys.stderr, flush=True)
                else:
                    print(f"[event] {msg_type}", flush=True)

            connection.on(EventType.OPEN, lambda _: print("Connection opened", flush=True))
            connection.on(EventType.MESSAGE, on_message)
            connection.on(EventType.CLOSE, lambda _: (print("Connection closed", flush=True), closed.set()))
            connection.on(EventType.ERROR, lambda error: print(f"Connection error: {error}", file=sys.stderr, flush=True))

            listen_task = asyncio.create_task(connection.start_listening())
            print(f"Synthesizing {len(text)} characters with {args.model}", flush=True)
            await connection.send_text(SpeakV1Text(text=text))
            # Per the streaming TTS contract, Close flushes buffered text and
            # gracefully closes after all generated audio is sent.
            await connection.send_close(SpeakV1Close(type="Close"))
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(closed.wait(), timeout=args.close_timeout)
            listen_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await listen_task

    if bytes_written == 0:
        raise RuntimeError(
            "No audio bytes were returned. Confirm this endpoint is a TTS endpoint "
            "and that the requested voice, encoding, and sample rate are supported."
        )

    print(f"Wrote {bytes_written} audio bytes to {args.output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthesize text with a Deepgram TTS SageMaker endpoint via the Deepgram Python SDK."
    )
    parser.add_argument("endpoint_name", help="SageMaker endpoint name")
    parser.add_argument("text", nargs="?", default=DEFAULT_TEXT, help="Text to synthesize")
    parser.add_argument("--text-file", type=Path, help="UTF-8 text file to synthesize")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tts-sdk-output.pcm"),
        help="Output audio path; linear16 streaming output is raw PCM by default",
    )
    parser.add_argument("--region", default=DEFAULT_REGION, help=f"AWS region (default: {DEFAULT_REGION})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Deepgram TTS voice model (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--encoding",
        choices=STREAMING_ENCODINGS,
        default="linear16",
        help="Streaming output encoding (default: linear16)",
    )
    parser.add_argument("--sample-rate", type=int, default=24000, help="Output sample rate (default: 24000)")
    parser.add_argument("--close-timeout", type=float, default=30.0, help="Seconds to wait for graceful close (default: 30)")
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    try:
        await synthesize(args)
    except Exception as exc:
        print(
            "TTS SDK smoke test failed. Check that the endpoint name and region are correct, "
            "AWS credentials allow sagemaker:InvokeEndpointWithBidirectionalStream, "
            "the endpoint is InService, and the requested TTS voice/encoding/sample rate "
            "are supported by the deployed model. "
            f"Original error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    asyncio.run(main())
