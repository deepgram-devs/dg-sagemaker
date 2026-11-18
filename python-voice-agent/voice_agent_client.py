#!/usr/bin/env python3
"""Talk to a Deepgram Voice Agent SageMaker endpoint.

One endpoint serves three surfaces:

  /v1/agent/converse  the Voice Agent: a WebSocket session over SageMaker
                      bidirectional streaming. Caller audio in, agent audio and
                      JSON events out. This file's `AgentSession`.
  /v2/listen          standalone Flux speech-to-text (WebSocket). `ListenStream`.
  /v2/speak           standalone Flux text-to-speech; here the batch form via
                      `invoke_endpoint`. `synthesize()`.

Everything runs inside the container: speech-to-text, the LLM bundled in the
container, and text-to-speech. The session's `Settings` point the agent's
`think` step at that bundled LLM (`http://127.0.0.1:3081/v1/chat/completions`,
model `phonellm`), so no request leaves the endpoint.

Example: hold a short conversation and save what the agent said.

    AWS_PROFILE=my-profile uv run voice_agent_client.py <endpoint> \\
        --say "Hi, do you have a table for four tomorrow at seven?" \\
        --out agent.wav
    uv run voice_agent_client.py <endpoint> --wav caller16k.wav --think-live

`--wav` must be 16 kHz mono 16-bit PCM. Audio is streamed at real time and
followed by silence, as a phone line would: end-of-turn is detected from the
audio itself, so a client that simply stops sending never ends its turn.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from urllib.parse import quote

from aws_sdk_sagemaker_runtime_http2.aio.client import AsyncSageMakerRuntimeHTTP2Client
from aws_sdk_sagemaker_runtime_http2.aio.config import AsyncSageMakerRuntimeHTTP2Config
from aws_sdk_sagemaker_runtime_http2.models import (
    InvokeEndpointWithBidirectionalStreamInput,
    RequestPayloadPart,
    RequestStreamEventPayloadPart,
)
from smithy_aws_core.aio.identity import AsyncEnvironmentCredentialsResolver
from smithy_http.aio.crt import AWSCRTHTTPClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e2e.e2e_test_common import (  # noqa: E402
    CHUNK_BYTES,
    CHUNK_MS,
    DEFAULT_VOICE,
    GREETING,
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    PROMPT,
    SILENCE_CHUNK,
    read_wav_16k_mono,
    strip_wav,
    write_wav,
)

logger = logging.getLogger(__name__)

AGENT_PATH = "v1/agent/converse"
LISTEN_PATH = "v2/listen"
LISTEN_MODEL = "flux-general-multi"

#: The LLM bundled in the container, reachable only from inside it.
BUNDLED_LLM_URL = "http://127.0.0.1:3081/v1/chat/completions"
BUNDLED_LLM_MODEL = "phonellm"

#: SageMaker's bidirectional-streaming port. NOT 443 — see `make_client`.
BIDI_PORT = 8443


# ---------------------------------------------------------------------------
# Transport (same shape as the other drivers in this repo)
# ---------------------------------------------------------------------------


async def _bidi_client(endpoint_uri: str, region: str, transport=None) -> AsyncSageMakerRuntimeHTTP2Client:
    """HTTP/2 SageMaker runtime client (aws-sdk-sagemaker-runtime-http2 0.13).

    The config is resolved up front with everything pinned: the endpoint URI
    (port 8443 is mandatory for the bidirectional stream), the region, the
    credentials (read from the environment; see `ensure_env_credentials`) and
    the AWS CRT HTTP/2 transport. The SDK's default transport is HTTP/1.1-only
    and cannot carry the stream.
    """
    config = await AsyncSageMakerRuntimeHTTP2Config.resolve(
        endpoint_uri=endpoint_uri,
        region=region,
        aws_credentials_identity_resolver=AsyncEnvironmentCredentialsResolver(),
        transport=transport or AWSCRTHTTPClient(),
    )
    return AsyncSageMakerRuntimeHTTP2Client(config=config)


def _event_payload(result) -> bytes | None:
    """Payload bytes of one response event, or None when there is nothing to handle.

    Server-side stream failures arrive as typed events with no `bytes_`; they are
    turned into a Deepgram-style `{"type": "Error", ...}` message so callers
    record them as errors instead of tripping on a missing attribute.
    """
    value = getattr(result, "value", None)
    if value is None:
        return None
    if hasattr(value, "bytes_"):
        return value.bytes_ or None
    msg = getattr(value, "message", None) or repr(value)
    code = getattr(value, "error_code", None)
    return json.dumps({"type": "Error", "description": f"{type(result).__name__}: {msg}"
                       + (f" (error_code={code})" if code else "")}).encode("utf-8")


def bidi_endpoint_uri(region: str, fips: bool = False) -> str:
    """The SageMaker runtime URI for bidirectional streaming (port 8443).

    `fips` selects the FIPS 140-3 runtime host; port 8443 is served there too.
    """
    host = "runtime-fips" if fips else "runtime"
    return f"https://{host}.sagemaker.{region}.amazonaws.com:{BIDI_PORT}"


async def make_client(region: str, fips: bool = False) -> AsyncSageMakerRuntimeHTTP2Client:
    """One HTTP/2 client. Port 8443 is mandatory: on the default 443 the call
    connects and then never answers, which looks exactly like a dead endpoint."""
    return await _bidi_client(bidi_endpoint_uri(region, fips), region)


class PerStreamClient:
    """Opens each bidirectional stream on its OWN HTTP/2 client.

    Several streams multiplexed on one connection starve each other (later
    streams never connect, earlier ones stall until the server drops them), so
    every session gets a connection of its own.
    """

    def __init__(self, region: str, fips: bool = False):
        self.region = region
        self.fips = fips
        self._clients: list[AsyncSageMakerRuntimeHTTP2Client] = []

    async def invoke_endpoint_with_bidirectional_stream(self, stream_input):
        client = await make_client(self.region, self.fips)
        self._clients.append(client)
        return await client.invoke_endpoint_with_bidirectional_stream(stream_input)

    async def close(self) -> None:
        for c in self._clients:
            try:
                await c.close()
            except Exception:  # noqa: BLE001
                pass
        self._clients.clear()


def ensure_env_credentials(region: str) -> None:
    """Materialise the active boto3 credentials into the process environment.

    The HTTP/2 client only reads environment credentials, so an
    `AWS_PROFILE`-based run would otherwise fail to sign. Idempotent.
    """
    import boto3

    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return
    creds = boto3.Session(region_name=region).get_credentials()
    if creds is None:
        raise SystemExit("no AWS credentials found — set AWS_PROFILE or the AWS_* env vars")
    frozen = creds.get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
    if frozen.token:
        os.environ["AWS_SESSION_TOKEN"] = frozen.token
    os.environ.setdefault("AWS_DEFAULT_REGION", region)


def runtime_client(region: str, fips: bool = False):
    """boto3 `sagemaker-runtime` client; `fips` applies to this client only."""
    import boto3
    from botocore.config import Config

    return boto3.Session(region_name=region).client(
        "sagemaker-runtime", config=Config(use_fips_endpoint=fips)
    )


# ---------------------------------------------------------------------------
# /v2/speak (batch) — used to synthesize caller audio from text
# ---------------------------------------------------------------------------


def synthesize(rt, endpoint: str, text: str, voice: str = DEFAULT_VOICE,
               sample_rate: int = INPUT_SAMPLE_RATE) -> bytes:
    """Linear16 PCM of `text` from the endpoint's own `/v2/speak`.

    The target path rides in `CustomAttributes`; `Content-Type:
    application/json` is required for the `{"text": ...}` body to be read.
    """
    q = f"model={voice}&encoding=linear16&sample_rate={sample_rate}&container=none"
    r = rt.invoke_endpoint(
        EndpointName=endpoint, ContentType="application/json", Accept="*/*",
        CustomAttributes=f"v2/speak?{q}", Body=json.dumps({"text": text}).encode("utf-8"),
    )
    pcm, _ = strip_wav(r["Body"].read())
    return pcm


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def agent_settings(*, voice: str = DEFAULT_VOICE, prompt: str = PROMPT,
                   greeting: str | None = GREETING, think_live: bool = False,
                   functions: list[dict] | None = None) -> dict:
    """A `Settings` message for this endpoint.

    `listen` is Flux (multilingual), `think` is the LLM bundled in the container,
    `speak` is a Flux voice. `think_live` turns on Conversational Mode, which
    adds backchannels and earlier, more natural turn-taking.
    """
    think: dict = {
        "provider": {"type": "open_ai", "model": BUNDLED_LLM_MODEL},
        "endpoint": {"url": BUNDLED_LLM_URL, "headers": {}},
        "prompt": prompt,
    }
    if functions:
        think["functions"] = functions
    agent: dict = {
        "listen": {"provider": {"type": "deepgram", "model": LISTEN_MODEL, "version": "v2"}},
        "think": think,
        "speak": {"provider": {"type": "deepgram", "version": "v2", "model": voice}},
    }
    if greeting:
        agent["greeting"] = greeting
    if think_live:
        agent["think_live"] = {"provider": {"type": "deepgram"}}
    return {
        "type": "Settings",
        "audio": {
            "input": {"encoding": "linear16", "sample_rate": INPUT_SAMPLE_RATE},
            "output": {"encoding": "linear16", "sample_rate": OUTPUT_SAMPLE_RATE, "container": "none"},
        },
        "agent": agent,
    }


# ---------------------------------------------------------------------------
# /v1/agent/converse
# ---------------------------------------------------------------------------


class AgentSession:
    """One Voice Agent session over SageMaker bidirectional streaming.

    Every server event is kept in `events` as `(monotonic_time, dict)`, and every
    agent audio frame in `audio` as `(monotonic_time, bytes)`, so callers can ask
    questions after the fact ("how much audio arrived after X?").

    A session that the endpoint refuses at start (for example an unknown query
    parameter) does not raise: `rejected` is set and the reason is in `errors`.
    """

    def __init__(self, client, endpoint: str, query: str = "", cid: int = 0):
        self.client = client
        self.endpoint = endpoint
        self.query = query
        self.cid = cid
        self.stream = None
        self.output_stream = None
        self.reader: asyncio.Task | None = None
        self.events: list[tuple[float, dict]] = []
        self.audio: list[tuple[float, bytes]] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.rejected = False
        self.closed = False
        self.opened_at: float | None = None
        self._changed = asyncio.Event()

    # -- lifecycle --------------------------------------------------------

    async def start(self, timeout: float = 60.0) -> bool:
        """Open the stream. Returns False (and records why) if it was refused."""
        kw = {"model_query_string": self.query} if self.query else {}
        # `model_query_string` is omitted when empty: SageMaker answers an empty
        # one with a 400 ValidationError.
        try:
            self.stream = await self.client.invoke_endpoint_with_bidirectional_stream(
                InvokeEndpointWithBidirectionalStreamInput(
                    endpoint_name=self.endpoint, model_invocation_path=AGENT_PATH, **kw))
            _, self.output_stream = await asyncio.wait_for(self.stream.await_output(), timeout)
        except Exception as e:  # noqa: BLE001 — a refused session arrives as an exception
            self.rejected = True
            self.errors.append(f"session start refused: {type(e).__name__}: {e}"[:300])
            return False
        self.opened_at = time.monotonic()
        self.reader = asyncio.create_task(self._read_loop())
        return True

    async def _read_loop(self) -> None:
        try:
            while True:
                result = await self.output_stream.receive()
                if result is None:
                    break
                payload = _event_payload(result)
                if payload:
                    self._handle(payload)
        except Exception as e:  # noqa: BLE001
            # The bidi transport raises on most teardowns; not a failure by itself.
            logger.debug("[%s] read loop ended: %s", self.cid, e)
        finally:
            self.closed = True
            self._changed.set()

    def _handle(self, data: bytes) -> None:
        now = time.monotonic()
        try:
            msg = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            msg = None
        if not isinstance(msg, dict) or "type" not in msg:
            self.audio.append((now, data))         # binary frame = agent audio
        else:
            self.events.append((now, msg))
            t = msg["type"]
            if t == "Error":
                self.errors.append(f"{msg.get('code', '')} {msg.get('description', '')}".strip())
            elif t == "Warning":
                self.warnings.append(f"{msg.get('code', '')} {msg.get('description', '')}".strip())
            logger.debug("[%s] %s %s", self.cid, t, str(msg)[:160])
        self._changed.set()

    async def finish(self) -> None:
        if self.stream is not None:
            try:
                await self.stream.input_stream.close()
            except Exception:  # noqa: BLE001
                pass
        if self.reader is not None:
            try:
                await asyncio.wait_for(self.reader, 10)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                self.reader.cancel()

    # -- sending ----------------------------------------------------------

    async def _send(self, data: bytes, kind: str) -> bool:
        if self.closed or self.stream is None:
            return False
        try:
            await self.stream.input_stream.send(
                RequestStreamEventPayloadPart(value=RequestPayloadPart(bytes_=data, data_type=kind)))
            return True
        except Exception as e:  # noqa: BLE001
            logger.debug("[%s] send failed: %s", self.cid, e)
            self.closed = True
            return False

    async def send_json(self, msg: dict) -> bool:
        return await self._send(json.dumps(msg).encode("utf-8"), "UTF8")

    async def send_audio(self, pcm: bytes) -> bool:
        return await self._send(pcm, "BINARY")

    async def speak(self, pcm: bytes) -> float:
        """Stream caller audio at real time. Returns the time the last frame went out."""
        for i in range(0, len(pcm), CHUNK_BYTES):
            if not await self.send_audio(pcm[i:i + CHUNK_BYTES]):
                break
            await asyncio.sleep(CHUNK_MS / 1000)
        return time.monotonic()

    async def wait(self, pred, timeout: float, *, feed_silence: bool = True) -> bool:
        """Wait until `pred()` is true, streaming silence meanwhile (a live call
        never stops sending audio). False on timeout or if the session closed."""
        deadline = time.monotonic() + timeout
        while not pred():
            if self.closed or time.monotonic() > deadline:
                return pred()
            if feed_silence:
                await self.send_audio(SILENCE_CHUNK)
                await asyncio.sleep(CHUNK_MS / 1000)
            else:
                self._changed.clear()
                try:
                    await asyncio.wait_for(self._changed.wait(), 0.25)
                except asyncio.TimeoutError:
                    pass
        return True

    # -- queries ----------------------------------------------------------

    def types(self, since: float = 0.0) -> list[str]:
        return [m["type"] for t, m in self.events if t >= since]

    def first(self, type_: str, since: float = 0.0, **match) -> tuple[float, dict] | None:
        for t, m in self.events:
            if t >= since and m.get("type") == type_ and all(m.get(k) == v for k, v in match.items()):
                return t, m
        return None

    def texts(self, role: str, since: float = 0.0) -> list[str]:
        return [m.get("content", "") for t, m in self.events
                if t >= since and m.get("type") == "ConversationText" and m.get("role") == role]

    def audio_between(self, start: float, end: float = float("inf")) -> bytes:
        return b"".join(b for t, b in self.audio if start <= t < end)

    def first_audio_after(self, since: float) -> float | None:
        return next((t for t, _ in self.audio if t >= since), None)

    def last_activity(self) -> float:
        """Time of the newest agent audio frame or agent-side event."""
        times = [t for t, _ in self.audio[-1:]]
        times += [t for t, m in self.events[-1:]]
        return max(times) if times else (self.opened_at or 0.0)

    async def wait_reply(self, since: float, timeout: float = 45.0, quiet_s: float = 1.5) -> bool:
        """Wait for the agent's reply to everything said since `since`.

        A reply can be several statements (a short acknowledgement, then the
        answer), each with its own `AgentAudioDone`, so this waits for an
        `AgentAudioDone` after `since` AND then `quiet_s` with no further agent
        audio or events.
        """
        def done() -> bool:
            hit = self.first("AgentAudioDone", since)
            return bool(hit) and time.monotonic() - self.last_activity() >= quiet_s
        return await self.wait(done, timeout)


# ---------------------------------------------------------------------------
# /v2/listen — standalone Flux speech-to-text on the same endpoint
# ---------------------------------------------------------------------------


async def transcribe(client, endpoint: str, pcm: bytes, *, model: str = LISTEN_MODEL,
                     timeout: float = 30.0) -> tuple[list[str], list[str]]:
    """Stream `pcm` (16 kHz linear16) to `/v2/listen` at real time; returns
    `(end_of_turn_transcripts, errors)`."""
    query = "&".join(f"{k}={quote(str(v))}" for k, v in
                     {"model": model, "encoding": "linear16", "sample_rate": INPUT_SAMPLE_RATE}.items())
    turns: list[str] = []
    errors: list[str] = []
    try:
        stream = await client.invoke_endpoint_with_bidirectional_stream(
            InvokeEndpointWithBidirectionalStreamInput(
                endpoint_name=endpoint, model_invocation_path=LISTEN_PATH, model_query_string=query))
        _, out = await asyncio.wait_for(stream.await_output(), 60)
    except Exception as e:  # noqa: BLE001
        return turns, [f"session start refused: {type(e).__name__}: {e}"[:300]]

    async def send(b: bytes, kind: str) -> None:
        await stream.input_stream.send(RequestStreamEventPayloadPart(value=RequestPayloadPart(bytes_=b, data_type=kind)))

    async def read() -> None:
        try:
            while True:
                r = await out.receive()
                if r is None:
                    return
                p = _event_payload(r)
                if not p:
                    continue
                try:
                    m = json.loads(p)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if m.get("type") == "TurnInfo" and m.get("event") == "EndOfTurn":
                    turns.append(m.get("transcript", ""))
                elif m.get("type") == "Error":
                    errors.append(f"{m.get('code', '')} {m.get('description', '')}".strip())
        except Exception as e:  # noqa: BLE001
            logger.debug("listen read loop ended: %s", e)

    reader = asyncio.create_task(read())
    try:
        tail = pcm + SILENCE_CHUNK * 15          # trailing silence lets the turn end
        for i in range(0, len(tail), CHUNK_BYTES):
            await send(tail[i:i + CHUNK_BYTES], "BINARY")
            await asyncio.sleep(CHUNK_MS / 1000)
        await send(json.dumps({"type": "CloseStream"}).encode(), "UTF8")
        await asyncio.wait_for(reader, timeout)
    except asyncio.TimeoutError:
        reader.cancel()
    except Exception as e:  # noqa: BLE001
        errors.append(f"{type(e).__name__}: {e}"[:200])
        reader.cancel()
    finally:
        try:
            await stream.input_stream.close()
        except Exception:  # noqa: BLE001
            pass
    return turns, errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def converse(args) -> int:
    ensure_env_credentials(args.region)
    rt = runtime_client(args.region, args.fips)
    if args.wav:
        caller = read_wav_16k_mono(args.wav)
    else:
        caller = synthesize(rt, args.endpoint, args.say, args.caller_voice)
    print(f"caller audio: {len(caller) / 2 / INPUT_SAMPLE_RATE:.1f}s", file=sys.stderr)

    client = PerStreamClient(args.region, args.fips)
    s = AgentSession(client, args.endpoint)
    t0 = time.monotonic()
    if not await s.start():
        print(f"session refused: {s.errors[0]}", file=sys.stderr)
        return 1
    printed = 0

    def show() -> None:
        nonlocal printed
        for t, m in s.events[printed:]:
            if m["type"] == "ConversationText":
                print(f"[{t - t0:6.2f}s] {m.get('role'):>9}: {m.get('content')}")
            elif m["type"] in ("Welcome", "SettingsApplied", "UserStartedSpeaking",
                               "AgentAudioDone", "Error", "Warning", "FunctionCallRequest"):
                print(f"[{t - t0:6.2f}s] {m['type']} {json.dumps({k: v for k, v in m.items() if k != 'type'})[:120]}")
        printed = len(s.events)

    try:
        await s.send_json(agent_settings(voice=args.voice, prompt=args.prompt,
                                         greeting=args.greeting or None, think_live=args.think_live))
        if args.greeting:
            await s.wait_reply(t0, timeout=60)
        show()
        mark = time.monotonic()
        speech_end = await s.speak(caller)
        ok = await s.wait_reply(mark, timeout=args.timeout)
        show()
        first = s.first_audio_after(speech_end)
        if first:
            print(f"speech end -> first agent audio: {(first - speech_end) * 1000:.0f} ms", file=sys.stderr)
    finally:
        await s.finish()
        await client.close()
    show()

    pcm = b"".join(b for _, b in s.audio)
    if args.out:
        write_wav(args.out, pcm, OUTPUT_SAMPLE_RATE)
        print(f"agent audio: {len(pcm) / 2 / OUTPUT_SAMPLE_RATE:.1f}s -> {args.out}", file=sys.stderr)
    if s.errors:
        print(f"errors: {s.errors}", file=sys.stderr)
    return 0 if ok and not s.errors else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("endpoint", help="SageMaker endpoint name")
    p.add_argument("--region", default="us-east-2")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--wav", help="caller audio: 16 kHz mono 16-bit WAV")
    src.add_argument("--say", default="Hi, do you have a table for four tomorrow evening at seven?",
                     help="caller line, synthesized with the endpoint's own text-to-speech (default)")
    p.add_argument("--voice", default=DEFAULT_VOICE, help=f"agent voice (default {DEFAULT_VOICE})")
    p.add_argument("--caller-voice", default="flux-alexis-en", help="voice used to synthesize --say")
    p.add_argument("--prompt", default=PROMPT, help="the agent's instructions")
    p.add_argument("--greeting", default=GREETING, help="first thing the agent says ('' for none)")
    p.add_argument("--think-live", action="store_true",
                   help="turn on Conversational Mode (backchannels, earlier turn-taking)")
    p.add_argument("--out", default="agent.wav", help="where to save the agent's audio ('' to skip)")
    p.add_argument("--timeout", type=float, default=45, help="seconds to wait for the reply")
    p.add_argument("--fips", action="store_true",
                   help="Route AWS calls to the FIPS 140-3 endpoints (bidi streaming on "
                        "runtime-fips.sagemaker.<region>.amazonaws.com:8443). OFF by default.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    print(f"Bidi URL:    {bidi_endpoint_uri(args.region, args.fips)}", file=sys.stderr)
    print(f"FIPS:        {'yes (--fips)' if args.fips else 'no'}", file=sys.stderr)
    return asyncio.run(converse(args))


if __name__ == "__main__":
    sys.exit(main())
