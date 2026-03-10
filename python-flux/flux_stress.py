#!/usr/bin/env python3
"""
Deepgram Flux STT SageMaker Stress Test

Streams audio to multiple simultaneous bidirectional connections to a Deepgram
Flux model deployed on SageMaker.  Supports two input modes:

  file        – Stream a WAV file at real-time pace (repeatable load testing).
  microphone  – Capture live microphone input via PyAudio.

Flux uses the /v2/listen endpoint and a turn-based message protocol with
integrated end-of-turn detection, replacing the /v1/listen channel/alternatives
format used by Nova models.

Supported Flux input messages:
  - Binary audio frames (raw PCM bytes)
  - Configure   (update thresholds/keyterms mid-stream)
  - KeepAlive   (prevent idle timeout)
  - CloseStream (gracefully terminate the stream)

Supported Flux output messages:
  - Connected        (stream established)
  - TurnInfo         (transcript updates; event field indicates state)
  - ConfigureSuccess (Configure acknowledged)
  - ConfigureFailure (Configure rejected)
  - Error            (fatal; connection terminated by server)
"""

import asyncio
import argparse
import json
import logging
import os
import signal
import sys
import time
import wave
from queue import Queue
from urllib.parse import quote
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from aws_sdk_sagemaker_runtime_http2.client import SageMakerRuntimeHTTP2Client
from aws_sdk_sagemaker_runtime_http2.config import Config, HTTPAuthSchemeResolver
from aws_sdk_sagemaker_runtime_http2.models import (
    InvokeEndpointWithBidirectionalStreamInput,
    ModelStreamError,
    RequestStreamEventPayloadPart,
    RequestPayloadPart
)
from smithy_aws_core.identity import EnvironmentCredentialsResolver
from smithy_aws_core.auth.sigv4 import SigV4AuthScheme

# Configuration constants
DEFAULT_REGION = "us-east-1"
DEFAULT_MODEL = "flux-general-en"
FLUX_API_PATH = "v2/listen"
DEFAULT_MIC_SAMPLE_RATE = 16000

# Deepgram recommends 80ms audio chunks for Flux
AUDIO_CHUNK_MS = 80

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection – Flux /v2/listen bidirectional stream
# ---------------------------------------------------------------------------

class DeepgramFluxConnection:
    """
    Represents a single bidirectional streaming connection to a Deepgram Flux
    model on SageMaker.

    Handles the Flux v2 protocol: binary audio input, JSON control messages,
    and TurnInfo-based transcript responses with integrated end-of-turn detection.
    """

    def __init__(self, connection_id: int, client: SageMakerRuntimeHTTP2Client, endpoint_name: str):
        self.connection_id = connection_id
        self.client = client
        self.endpoint_name = endpoint_name
        self.stream = None
        self.output_stream = None
        self.is_active = False
        self.response_task = None
        self.chunk_count = 0
        self.turn_index = 0
        self.transcript_parts: list[str] = []
        self.close_requested = False

    async def start_session(
        self,
        sample_rate: int,
        model: str = DEFAULT_MODEL,
        encoding: str = "linear16",
        eot_threshold: float | None = None,
        eager_eot_threshold: float | None = None,
        eot_timeout_ms: int | None = None,
        keyterms: list[str] | None = None,
    ):
        """
        Open a Flux /v2/listen bidirectional stream on SageMaker.

        Args:
            sample_rate: Sample rate of the audio (e.g. 16000).
            model: Flux model name (default: flux-general-en).
            encoding: Audio encoding (default: linear16).
            eot_threshold: Confidence threshold for EndOfTurn (0.5–0.9).
            eager_eot_threshold: Confidence for EagerEndOfTurn; must be ≤ eot_threshold.
            eot_timeout_ms: Max silence ms before forced EndOfTurn (500–10000).
            keyterms: List of keyterms for boosting recognition accuracy.
        """
        query_params: dict[str, str | int | float] = {
            "model": model,
            "encoding": encoding,
            "sample_rate": sample_rate,
        }
        if eot_threshold is not None:
            query_params["eot_threshold"] = eot_threshold
        if eager_eot_threshold is not None:
            query_params["eager_eot_threshold"] = eager_eot_threshold
        if eot_timeout_ms is not None:
            query_params["eot_timeout_ms"] = eot_timeout_ms

        query_string = "&".join(f"{k}={v}" for k, v in query_params.items())

        if keyterms:
            keyterm_params = "&".join(f"keyterm={quote(kt)}" for kt in keyterms)
            query_string = f"{query_string}&{keyterm_params}"

        logger.debug(f"[Connection {self.connection_id}] Starting Flux session: {query_string}")

        stream_input = InvokeEndpointWithBidirectionalStreamInput(
            endpoint_name=self.endpoint_name,
            model_invocation_path=FLUX_API_PATH,
            model_query_string=query_string,
        )

        self.stream = await self.client.invoke_endpoint_with_bidirectional_stream(stream_input)
        self.is_active = True

        output = await self.stream.await_output()
        self.output_stream = output[1]

        self.response_task = asyncio.create_task(self._process_responses())
        await asyncio.sleep(0.1)

        logger.info(f"[Connection {self.connection_id}] Session started")

    # -------------------------------------------------------------------------
    # Input messages
    # -------------------------------------------------------------------------

    async def send_audio_chunk(self, audio_bytes: bytes):
        """Send a raw PCM audio frame to the stream."""
        if not self.is_active:
            return
        try:
            payload = RequestPayloadPart(bytes_=audio_bytes)
            event = RequestStreamEventPayloadPart(value=payload)
            await self.stream.input_stream.send(event)
            self.chunk_count += 1
            logger.debug(
                f"[Connection {self.connection_id}] Sent chunk {self.chunk_count} "
                f"({len(audio_bytes)} bytes)"
            )
        except Exception as e:
            logger.error(f"[Connection {self.connection_id}] Error sending audio chunk: {e}")
            self.is_active = False

    async def send_configure(
        self,
        eot_threshold: float | None = None,
        eager_eot_threshold: float | None = None,
        eot_timeout_ms: int | None = None,
        keyterms: list[str] | None = None,
    ):
        """
        Send a Configure message to update thresholds or keyterms mid-stream.

        Args:
            eot_threshold: New EndOfTurn confidence threshold.
            eager_eot_threshold: New EagerEndOfTurn confidence threshold.
            eot_timeout_ms: New silence timeout in milliseconds.
            keyterms: Replacement list of keyterms (replaces existing list).
        """
        if not self.is_active:
            return
        message: dict = {"type": "Configure"}
        thresholds = {}
        if eot_threshold is not None:
            thresholds["eot_threshold"] = eot_threshold
        if eager_eot_threshold is not None:
            thresholds["eager_eot_threshold"] = eager_eot_threshold
        if eot_timeout_ms is not None:
            thresholds["eot_timeout_ms"] = eot_timeout_ms
        if thresholds:
            message["thresholds"] = thresholds
        if keyterms is not None:
            message["keyterms"] = keyterms

        await self._send_json(message)
        logger.debug(f"[Connection {self.connection_id}] Sent Configure: {message}")

    async def send_keep_alive(self):
        """Send a KeepAlive message to prevent idle timeout."""
        if not self.is_active:
            return
        await self._send_json({"type": "KeepAlive"})
        logger.debug(f"[Connection {self.connection_id}] Sent KeepAlive")

    async def send_finalize(self):
        """
        Send a Finalize message to flush any buffered audio and complete the
        current turn immediately, regardless of end-of-turn detection state.
        """
        if not self.is_active:
            return
        await self._send_json({"type": "Finalize"})
        logger.debug(f"[Connection {self.connection_id}] Sent Finalize")

    async def send_close_stream(self):
        """Send a CloseStream message to gracefully terminate the stream."""
        if not self.is_active:
            return
        self.close_requested = True
        await self._send_json({"type": "CloseStream"})
        logger.debug(f"[Connection {self.connection_id}] Sent CloseStream")

    async def _send_json(self, message: dict):
        """Encode a JSON control message and write it to the input stream."""
        try:
            message_bytes = json.dumps(message).encode("utf-8")
            payload = RequestPayloadPart(bytes_=message_bytes, data_type="UTF8")
            event = RequestStreamEventPayloadPart(value=payload)
            await self.stream.input_stream.send(event)
        except Exception as e:
            logger.error(
                f"[Connection {self.connection_id}] Error sending {message.get('type')} message: {e}"
            )
            self.is_active = False

    # -------------------------------------------------------------------------
    # Output message processing
    # -------------------------------------------------------------------------

    async def _process_responses(self):
        """Receive and dispatch Flux /v2/listen server messages."""
        try:
            logger.debug(f"[Connection {self.connection_id}] Response processor started")

            while self.is_active:
                result = await self.output_stream.receive()
                if result is None:
                    logger.debug(f"[Connection {self.connection_id}] Stream closed by server")
                    break
                if result.value and result.value.bytes_:
                    self._handle_message(result.value.bytes_.decode("utf-8"))

            # Drain any remaining buffered responses after is_active goes False
            logger.debug(f"[Connection {self.connection_id}] Draining remaining responses...")
            drain_count = 0
            while drain_count < 20:
                try:
                    result = await asyncio.wait_for(self.output_stream.receive(), timeout=0.5)
                    if result is None:
                        break
                    if result.value and result.value.bytes_:
                        drain_count += 1
                        self._handle_message(result.value.bytes_.decode("utf-8"))
                except asyncio.TimeoutError:
                    break

            if drain_count:
                logger.debug(
                    f"[Connection {self.connection_id}] Drained {drain_count} buffered response(s)"
                )

        except ModelStreamError as e:
            if self.close_requested:
                logger.info(
                    f"[Connection {self.connection_id}] Stream closed after CloseStream: {e}"
                )
            else:
                logger.error(
                    f"[Connection {self.connection_id}] Error in response processor: {e}",
                    exc_info=True,
                )
        except Exception as e:
            logger.error(
                f"[Connection {self.connection_id}] Error in response processor: {e}",
                exc_info=True,
            )

    def _handle_message(self, raw: str):
        """Parse and dispatch a single server-side Flux JSON message."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"[Connection {self.connection_id}] Non-JSON response: {raw!r}")
            return

        msg_type = msg.get("type")

        if msg_type == "Connected":
            request_id = msg.get("request_id", "")
            seq = msg.get("sequence_id", 0)
            logger.info(
                f"[Connection {self.connection_id}] Connected "
                f"(request_id={request_id}, sequence_id={seq})"
            )

        elif msg_type == "TurnInfo":
            self._handle_turn_info(msg)

        elif msg_type == "ConfigureSuccess":
            thresholds = msg.get("thresholds", {})
            keyterms = msg.get("keyterms", [])
            logger.info(
                f"[Connection {self.connection_id}] ConfigureSuccess "
                f"(thresholds={thresholds}, keyterms={keyterms})"
            )

        elif msg_type == "ConfigureFailure":
            logger.warning(
                f"[Connection {self.connection_id}] ConfigureFailure — "
                "check that eager_eot_threshold ≤ eot_threshold"
            )

        elif msg_type == "Error":
            code = msg.get("code", "unknown")
            desc = msg.get("description", "")
            logger.error(
                f"[Connection {self.connection_id}] Fatal server error [{code}]: {desc}"
            )
            self.is_active = False

        else:
            logger.debug(f"[Connection {self.connection_id}] Unhandled message type: {msg_type}")

    def _handle_turn_info(self, msg: dict):
        """
        Process a TurnInfo message and print transcript output.

        Flux TurnInfo event types:
          Update          – Periodic transcript update (~every 250ms), no state change.
          StartOfTurn     – User has started speaking; transcript may be empty.
          EagerEndOfTurn  – High likelihood the user has finished; non-empty transcript;
                            use for speculative LLM pre-processing.
          TurnResumed     – User resumed speaking after EagerEndOfTurn; discard speculative work.
          EndOfTurn       – Turn complete; transcript is final and matches last EagerEndOfTurn.
        """
        event = msg.get("event", "")
        transcript = msg.get("transcript", "")
        turn_index = msg.get("turn_index", self.turn_index)
        seq = msg.get("sequence_id", "?")
        audio_start = msg.get("audio_window_start", 0.0)
        audio_end = msg.get("audio_window_end", 0.0)
        eot_confidence = msg.get("end_of_turn_confidence")

        logger.debug(
            f"[Connection {self.connection_id}] TurnInfo event={event} "
            f"turn={turn_index} seq={seq} "
            f"audio=[{audio_start:.2f}s-{audio_end:.2f}s]"
        )

        if event == "Connected":
            # Flux sometimes sends a Connected-like event via TurnInfo on reconnects
            logger.info(f"[Connection {self.connection_id}] Stream ready")

        elif event == "StartOfTurn":
            self.transcript_parts = []
            logger.debug(f"[Connection {self.connection_id}] Turn {turn_index} started")

        elif event == "Update":
            if transcript.strip():
                print(
                    f"[Conn {self.connection_id}]   {transcript} [update]"
                )

        elif event == "EagerEndOfTurn":
            eot_str = f" ({eot_confidence:.1%})" if eot_confidence is not None else ""
            print(
                f"[Conn {self.connection_id}] ~ {transcript}{eot_str} [eager, turn {turn_index}]"
            )

        elif event == "TurnResumed":
            print(
                f"[Conn {self.connection_id}]   ... resumed [turn {turn_index}]"
            )

        elif event == "EndOfTurn":
            eot_str = f" ({eot_confidence:.1%})" if eot_confidence is not None else ""
            if transcript.strip():
                print(
                    f"[Conn {self.connection_id}] ✓ {transcript}{eot_str} [turn {turn_index}]"
                )
            self.turn_index = turn_index + 1

        else:
            if transcript.strip():
                print(f"[Conn {self.connection_id}] [{event}] {transcript}")

    # -------------------------------------------------------------------------
    # Session teardown
    # -------------------------------------------------------------------------

    async def end_session(self):
        """
        Gracefully close the Flux stream.

        Sends Finalize (to flush buffered audio) then CloseStream before
        closing the underlying transport and waiting for the response task.
        """
        if not self.is_active:
            return

        logger.debug(f"[Connection {self.connection_id}] Ending session")

        # CloseStream tells Deepgram no more data is coming
        await self.send_close_stream()
        self.is_active = False

        try:
            await self.stream.input_stream.close()
        except Exception as e:
            logger.error(f"[Connection {self.connection_id}] Error closing input stream: {e}")

        if self.response_task and not self.response_task.done():
            try:
                await asyncio.wait_for(self.response_task, timeout=15.0)
                logger.debug(f"[Connection {self.connection_id}] All responses received")
            except asyncio.TimeoutError:
                logger.warning(
                    f"[Connection {self.connection_id}] Timeout waiting for final responses (15s)"
                )
                self.response_task.cancel()
            except asyncio.CancelledError:
                pass

        logger.info(
            f"[Connection {self.connection_id}] Session ended "
            f"(sent {self.chunk_count} audio chunks, {self.turn_index} turn(s) completed)"
        )


# ---------------------------------------------------------------------------
# Shared base client
# ---------------------------------------------------------------------------

class BaseFluxClient:
    """
    Shared infrastructure for multi-connection Flux streaming clients.

    Subclasses implement the audio source (WAV file or microphone) and call
    initialize_connections() once self.sample_rate is populated.
    """

    def __init__(
        self,
        endpoint_name: str,
        region: str = DEFAULT_REGION,
        num_connections: int = 1,
    ):
        self.endpoint_name = endpoint_name
        self.region = region
        self.num_connections = num_connections
        self.bidi_endpoint = f"https://runtime.sagemaker.{region}.amazonaws.com:8443"
        self.client: SageMakerRuntimeHTTP2Client | None = None
        self.connections: list[DeepgramFluxConnection] = []
        self.is_active = False
        self.sample_rate: int | None = None  # must be set before initialize_connections()

    def _initialize_client(self):
        """Resolve AWS credentials and build the SageMaker Runtime HTTP/2 client."""
        logger.debug("Initializing SageMaker client")

        try:
            session = boto3.Session(region_name=self.region)
            credentials = session.get_credentials()

            if credentials is None:
                raise NoCredentialsError()

            frozen = credentials.get_frozen_credentials()
            os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
            os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
            if frozen.token:
                os.environ["AWS_SESSION_TOKEN"] = frozen.token

            caller_identity = session.client("sts").get_caller_identity()
            logger.debug(f"Authenticated as: {caller_identity.get('Arn', 'Unknown')}")

        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error("AWS credentials not found. Configure via one of:")
            logger.error("  1. aws configure")
            logger.error("  2. Environment variables: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY")
            logger.error("  3. ~/.aws/credentials")
            logger.error("  4. IAM role (when running on AWS infrastructure)")
            raise RuntimeError("AWS credentials not available") from e

        config = Config(
            endpoint_uri=self.bidi_endpoint,
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            auth_scheme_resolver=HTTPAuthSchemeResolver(),
            auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="sagemaker")},
        )
        self.client = SageMakerRuntimeHTTP2Client(config=config)
        logger.info("SageMaker client initialized")

    async def initialize_connections(
        self,
        model: str = DEFAULT_MODEL,
        eot_threshold: float | None = None,
        eager_eot_threshold: float | None = None,
        eot_timeout_ms: int | None = None,
        keyterms: list[str] | None = None,
    ):
        """
        Start all bidirectional Flux streaming connections in parallel.

        Args:
            model: Flux model variant (default: flux-general-en).
            eot_threshold: EndOfTurn confidence threshold (0.5–0.9).
            eager_eot_threshold: EagerEndOfTurn threshold; must be ≤ eot_threshold.
            eot_timeout_ms: Silence timeout ms before forced EndOfTurn (500–10000).
            keyterms: Keyterms for recognition boosting.
        """
        if self.sample_rate is None:
            raise RuntimeError("sample_rate must be set before calling initialize_connections()")

        if not self.client:
            self._initialize_client()

        logger.info(f"Initializing {self.num_connections} connection(s)...")

        for i in range(self.num_connections):
            conn = DeepgramFluxConnection(i + 1, self.client, self.endpoint_name)
            self.connections.append(conn)

        await asyncio.gather(*[
            conn.start_session(
                sample_rate=self.sample_rate,
                model=model,
                eot_threshold=eot_threshold,
                eager_eot_threshold=eager_eot_threshold,
                eot_timeout_ms=eot_timeout_ms,
                keyterms=keyterms or [],
            )
            for conn in self.connections
        ])

        self.is_active = True
        logger.info(f"All {self.num_connections} connection(s) ready")

    async def end_all_sessions(self):
        """Close all streaming sessions gracefully."""
        if not self.is_active and not self.connections:
            return

        logger.debug("Ending all sessions")
        self.is_active = False

        logger.info(f"Closing {len(self.connections)} connection(s)...")
        await asyncio.gather(*[c.end_session() for c in self.connections])
        logger.info("All sessions ended")


# ---------------------------------------------------------------------------
# File client
# ---------------------------------------------------------------------------

class FileFluxClient(BaseFluxClient):
    """
    Streams a WAV audio file in real-time to multiple simultaneous Deepgram
    Flux connections on AWS SageMaker.

    Audio is paced to the WAV file's sample rate so it arrives at the endpoint
    at the same cadence as live microphone input. The file can be looped for
    extended stress test runs. Audio chunks are sized to the Deepgram-recommended
    80ms window for optimal Flux performance.
    """

    def __init__(
        self,
        endpoint_name: str,
        wav_path: str,
        region: str = DEFAULT_REGION,
        num_connections: int = 1,
    ):
        super().__init__(endpoint_name, region, num_connections)
        self.wav_path = wav_path
        self.channels: int | None = None
        self.sample_width: int | None = None
        self.duration_seconds: float | None = None

    def open_wav(self) -> wave.Wave_read:
        """Open and validate the WAV file, populating audio metadata."""
        try:
            wf = wave.open(self.wav_path, "rb")
        except FileNotFoundError:
            logger.error(f"WAV file not found: {self.wav_path}")
            raise
        except wave.Error as e:
            logger.error(f"Invalid WAV file '{self.wav_path}': {e}")
            raise

        self.sample_rate = wf.getframerate()
        self.channels = wf.getnchannels()
        self.sample_width = wf.getsampwidth()
        total_frames = wf.getnframes()
        self.duration_seconds = total_frames / self.sample_rate

        if self.sample_width != 2:
            wf.close()
            raise ValueError(
                f"WAV file must be 16-bit PCM (sample width 2 bytes). "
                f"Got {self.sample_width * 8}-bit audio. "
                "Convert with: ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 output.wav"
            )

        logger.info(
            f"WAV: {self.wav_path} | {self.sample_rate} Hz | "
            f"{self.channels}ch | {self.duration_seconds:.2f}s"
        )
        return wf

    async def stream_wav_audio(self, loop: bool = False):
        """
        Read the WAV file and broadcast 80ms audio chunks to all connections at
        real-time pace.

        Args:
            loop: Restart from the beginning when the file ends (until is_active=False).
        """
        frames_per_chunk = int(self.sample_rate * AUDIO_CHUNK_MS / 1000)
        bytes_per_frame = self.sample_width * self.channels
        chunk_duration = frames_per_chunk / self.sample_rate  # seconds

        play_count = 0
        total_chunks = 0

        while self.is_active:
            wf = self.open_wav()
            play_count += 1
            logger.info(f"Streaming WAV file (pass {play_count})...")

            chunk_start = time.monotonic()

            while self.is_active:
                raw = wf.readframes(frames_per_chunk)
                if not raw:
                    break  # End of file

                active_connections = [c for c in self.connections if c.is_active]
                if not active_connections:
                    logger.warning("All connections have failed")
                    self.is_active = False
                    break

                await asyncio.gather(*[c.send_audio_chunk(raw) for c in active_connections])
                total_chunks += 1

                # Pace delivery to real-time speed
                elapsed = time.monotonic() - chunk_start
                expected = total_chunks * chunk_duration
                sleep_time = expected - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            wf.close()
            logger.debug(f"End of WAV file (pass {play_count})")

            if not loop:
                break

        logger.info(f"Finished streaming: {play_count} pass(es), {total_chunks} chunks total")


# ---------------------------------------------------------------------------
# Microphone client
# ---------------------------------------------------------------------------

class MicFluxClient(BaseFluxClient):
    """
    Captures live microphone audio via PyAudio and streams it to multiple
    simultaneous Deepgram Flux connections on AWS SageMaker.

    Audio is captured in 80ms chunks (matching Deepgram's recommended frame
    size) and broadcast in real-time to all active connections.
    """

    def __init__(
        self,
        endpoint_name: str,
        region: str = DEFAULT_REGION,
        num_connections: int = 1,
        sample_rate: int = DEFAULT_MIC_SAMPLE_RATE,
        device_index: int | None = None,
    ):
        super().__init__(endpoint_name, region, num_connections)
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.audio_queue: Queue = Queue()
        self.pyaudio_instance = None
        self.audio_stream = None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback — enqueue captured audio for async broadcast."""
        try:
            import pyaudio
        except ImportError:
            pass
        if status:
            logger.warning(f"PyAudio stream status: {status}")
        self.audio_queue.put(in_data)
        try:
            import pyaudio as _pa
            return (None, _pa.paContinue)
        except ImportError:
            return (None, 0)

    async def start_microphone(self):
        """Initialize PyAudio and begin capturing microphone input."""
        try:
            import pyaudio
        except ImportError:
            print("ERROR: pyaudio is required for microphone input.")
            print("Install it with: uv add pyaudio")
            print("On macOS you may also need: brew install portaudio")
            raise

        logger.info("Initializing microphone")
        self.pyaudio_instance = pyaudio.PyAudio()

        logger.debug("Available audio input devices:")
        for i in range(self.pyaudio_instance.get_device_count()):
            info = self.pyaudio_instance.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                marker = " *" if self.device_index is not None and i == self.device_index else ""
                logger.debug(f"  [{i}] {info['name']}{marker}")

        frames_per_chunk = int(self.sample_rate * AUDIO_CHUNK_MS / 1000)

        open_kwargs: dict = dict(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=frames_per_chunk,
            stream_callback=self._audio_callback,
        )
        if self.device_index is not None:
            open_kwargs["input_device_index"] = self.device_index

        try:
            self.audio_stream = self.pyaudio_instance.open(**open_kwargs)
            self.audio_stream.start_stream()
            logger.info("Microphone started — speak now!")
        except Exception as e:
            logger.error(
                f"Failed to open microphone: {e}. "
                "Try listing devices with --list-devices or specifying --device INDEX."
            )
            raise

    async def stream_microphone_audio(self):
        """Broadcast audio chunks from the PyAudio queue to all active connections."""
        chunk_count = 0
        try:
            while self.is_active:
                if not self.audio_queue.empty():
                    audio_chunk = self.audio_queue.get()
                    chunk_count += 1

                    active = [c for c in self.connections if c.is_active]
                    if not active:
                        logger.warning("All connections have failed")
                        self.is_active = False
                        break

                    await asyncio.gather(*[c.send_audio_chunk(audio_chunk) for c in active])
                    logger.debug(f"Broadcast chunk {chunk_count} to {len(active)} connection(s)")
                else:
                    await asyncio.sleep(0.005)
        except Exception as e:
            logger.error(f"Error broadcasting microphone audio: {e}")
            raise
        finally:
            logger.info(f"Broadcasted {chunk_count} microphone audio chunks total")

    def stop_microphone(self):
        """Stop and release PyAudio resources."""
        if self.audio_stream:
            logger.debug("Stopping microphone stream")
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.pyaudio_instance:
            logger.debug("Terminating PyAudio")
            self.pyaudio_instance.terminate()
        logger.info("Microphone stopped")

    async def end_all_sessions(self):
        """Stop microphone then close all Flux sessions."""
        self.stop_microphone()
        await super().end_all_sessions()


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _add_common_args(parser: argparse.ArgumentParser):
    """Add arguments shared by both the file and microphone subcommands."""
    parser.add_argument("endpoint_name", help="SageMaker endpoint name")
    parser.add_argument(
        "--connections",
        type=int,
        default=1,
        help="Number of simultaneous Flux connections (default: 1)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Flux model variant (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--eot-threshold",
        type=float,
        default=None,
        metavar="0.5-0.9",
        help="EndOfTurn confidence threshold (default: Flux server default of 0.7)",
    )
    parser.add_argument(
        "--eager-eot-threshold",
        type=float,
        default=None,
        metavar="0.3-0.9",
        help=(
            "EagerEndOfTurn confidence threshold; enables eager events when set. "
            "Must be ≤ --eot-threshold."
        ),
    )
    parser.add_argument(
        "--eot-timeout-ms",
        type=int,
        default=None,
        metavar="500-10000",
        help="Max silence (ms) before a forced EndOfTurn (default: 5000)",
    )
    parser.add_argument(
        "--keyterms",
        default="",
        metavar="TERM1,TERM2",
        help="Comma-separated list of keyterms to boost recognition accuracy",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION})",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop automatically after this many seconds (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO)",
    )


def _validate_common_args(args) -> str | None:
    """Return an error message string, or None if arguments are valid."""
    if args.connections < 1:
        return "--connections must be a positive integer (minimum 1)"
    if args.eot_threshold is not None and not (0.5 <= args.eot_threshold <= 0.9):
        return "--eot-threshold must be between 0.5 and 0.9"
    if args.eager_eot_threshold is not None and not (0.3 <= args.eager_eot_threshold <= 0.9):
        return "--eager-eot-threshold must be between 0.3 and 0.9"
    if (
        args.eot_threshold is not None
        and args.eager_eot_threshold is not None
        and args.eager_eot_threshold > args.eot_threshold
    ):
        return "--eager-eot-threshold must be ≤ --eot-threshold"
    return None


def _print_banner(args, extra_lines: list[str]):
    print("=" * 60)
    print("Deepgram Flux STT SageMaker Stress Test")
    print("=" * 60)
    print(f"Endpoint:      {args.endpoint_name}")
    for line in extra_lines:
        print(line)
    print(f"Connections:   {args.connections}")
    print(f"Model:         {args.model}")
    print(f"Region:        {args.region}")
    print(f"Chunk Size:    {AUDIO_CHUNK_MS}ms")
    if args.eot_threshold is not None:
        print(f"EoT Threshold: {args.eot_threshold}")
    if args.eager_eot_threshold is not None:
        print(f"Eager EoT:     {args.eager_eot_threshold}")
    if args.eot_timeout_ms is not None:
        print(f"EoT Timeout:   {args.eot_timeout_ms}ms")
    keyterms = [kt.strip() for kt in args.keyterms.split(",") if kt.strip()] if args.keyterms else []
    if keyterms:
        print(f"Keyterms:      {', '.join(keyterms)}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

async def run_file(args) -> int:
    keyterms = [kt.strip() for kt in args.keyterms.split(",") if kt.strip()] if args.keyterms else []

    client = FileFluxClient(
        endpoint_name=args.endpoint_name,
        wav_path=args.file,
        region=args.region,
        num_connections=args.connections,
    )

    # Validate WAV file early, before printing the banner
    try:
        wf = client.open_wav()
        wf.close()
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    limit_str = (
        f"{args.duration}s"
        if args.duration
        else ("until file ends (looping)" if args.loop else "until file ends or Ctrl+C")
    )

    _print_banner(args, [
        f"Input:         file ({args.file})",
        f"File Duration: {client.duration_seconds:.2f}s",
        f"Sample Rate:   {client.sample_rate} Hz",
        f"Channels:      {client.channels}",
        f"Loop:          {'yes' if args.loop else 'no'}",
        f"Limit:         {limit_str}",
    ])

    def signal_handler(sig, frame):
        print("\n\nReceived interrupt signal, stopping...")
        client.is_active = False

    signal.signal(signal.SIGINT, signal_handler)

    try:
        await client.initialize_connections(
            model=args.model,
            eot_threshold=args.eot_threshold,
            eager_eot_threshold=args.eager_eot_threshold,
            eot_timeout_ms=args.eot_timeout_ms,
            keyterms=keyterms,
        )

        print("\n" + "=" * 60)
        print(f"LIVE TRANSCRIPTION - {args.connections} Flux Connection(s)")
        print("  ✓  = EndOfTurn  |  ~  = EagerEndOfTurn  |  [update] = interim")
        if args.duration:
            print(f"   (Running for {args.duration}s, or press Ctrl+C to stop early)")
        else:
            print("   (Press Ctrl+C to stop)")
        print("=" * 60 + "\n")

        stream_coro = client.stream_wav_audio(loop=args.loop)

        if args.duration:
            try:
                await asyncio.wait_for(stream_coro, timeout=args.duration)
            except asyncio.TimeoutError:
                print(f"\nDuration of {args.duration}s reached, stopping...")
                client.is_active = False
        else:
            await stream_coro

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during streaming: {e}", exc_info=True)
        return 1
    finally:
        await client.end_all_sessions()
        print("\n" + "=" * 60)

    logger.info("Stress test complete")
    return 0


async def run_microphone(args) -> int:
    # Check for pyaudio before doing anything else
    try:
        import pyaudio  # noqa: F401
    except ImportError:
        print("ERROR: pyaudio is required for microphone input.")
        print("Install it with: uv add pyaudio")
        print("On macOS you may also need: brew install portaudio")
        return 1

    # Handle --list-devices flag
    if args.list_devices:
        import pyaudio
        pa = pyaudio.PyAudio()
        print("Available audio input devices:")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                print(f"  [{i}] {info['name']} ({int(info['defaultSampleRate'])} Hz)")
        pa.terminate()
        return 0

    keyterms = [kt.strip() for kt in args.keyterms.split(",") if kt.strip()] if args.keyterms else []

    client = MicFluxClient(
        endpoint_name=args.endpoint_name,
        region=args.region,
        num_connections=args.connections,
        sample_rate=args.sample_rate,
        device_index=args.device,
    )

    limit_str = f"{args.duration}s" if args.duration else "until Ctrl+C"

    _print_banner(args, [
        f"Input:         microphone",
        f"Sample Rate:   {args.sample_rate} Hz",
        f"Device:        {args.device if args.device is not None else 'default'}",
        f"Limit:         {limit_str}",
    ])

    def signal_handler(sig, frame):
        print("\n\nReceived interrupt signal, stopping...")
        client.is_active = False

    signal.signal(signal.SIGINT, signal_handler)

    try:
        await client.initialize_connections(
            model=args.model,
            eot_threshold=args.eot_threshold,
            eager_eot_threshold=args.eager_eot_threshold,
            eot_timeout_ms=args.eot_timeout_ms,
            keyterms=keyterms,
        )

        await client.start_microphone()

        print("\n" + "=" * 60)
        print(f"LIVE TRANSCRIPTION - {args.connections} Flux Connection(s)")
        print("  ✓  = EndOfTurn  |  ~  = EagerEndOfTurn  |  [update] = interim")
        if args.duration:
            print(f"   (Running for {args.duration}s, or press Ctrl+C to stop early)")
        else:
            print("   (Press Ctrl+C to stop)")
        print("=" * 60 + "\n")

        stream_coro = client.stream_microphone_audio()

        if args.duration:
            try:
                await asyncio.wait_for(stream_coro, timeout=args.duration)
            except asyncio.TimeoutError:
                print(f"\nDuration of {args.duration}s reached, stopping...")
                client.is_active = False
        else:
            await stream_coro

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during streaming: {e}", exc_info=True)
        return 1
    finally:
        await client.end_all_sessions()
        print("\n" + "=" * 60)

    logger.info("Stress test complete")
    return 0


# ---------------------------------------------------------------------------
# list-endpoints subcommand handler
# ---------------------------------------------------------------------------

# Maps SageMaker endpoint status values to a short display label and colour code.
_STATUS_DISPLAY = {
    "InService":      ("InService",      "\033[32m"),   # green
    "Creating":       ("Creating",       "\033[33m"),   # yellow
    "Updating":       ("Updating",       "\033[33m"),   # yellow
    "RollingBack":    ("RollingBack",    "\033[33m"),   # yellow
    "SystemUpdating": ("SystemUpdating", "\033[33m"),   # yellow
    "Failed":         ("Failed",         "\033[31m"),   # red
    "Deleting":       ("Deleting",       "\033[31m"),   # red
    "OutOfService":   ("OutOfService",   "\033[90m"),   # grey
}
_RESET = "\033[0m"


def _colour_status(status: str, use_colour: bool) -> str:
    label, code = _STATUS_DISPLAY.get(status, (status, "\033[0m"))
    return f"{code}{label}{_RESET}" if use_colour else label


def run_list_endpoints(args) -> int:
    """List SageMaker endpoints, optionally filtered by status."""
    try:
        session = boto3.Session(region_name=args.region)
        sm = session.client("sagemaker")
    except (NoCredentialsError, PartialCredentialsError) as e:
        print("ERROR: AWS credentials not found. Configure via one of:")
        print("  1. aws configure")
        print("  2. Environment variables: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY")
        print("  3. ~/.aws/credentials")
        print("  4. IAM role (when running on AWS infrastructure)")
        return 1

    status_filter = args.status.capitalize() if args.status else None

    try:
        list_kwargs: dict = {}
        if status_filter:
            list_kwargs["StatusEquals"] = status_filter

        endpoints: list[dict] = []
        paginator = sm.get_paginator("list_endpoints")
        for page in paginator.paginate(**list_kwargs):
            endpoints.extend(page.get("Endpoints", []))

    except sm.exceptions.ClientError as e:
        print(f"ERROR: Failed to list endpoints: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: Unexpected error listing endpoints: {e}")
        logger.debug("list_endpoints error", exc_info=True)
        return 1

    use_colour = sys.stdout.isatty()

    if not endpoints:
        qualifier = f" with status '{status_filter}'" if status_filter else ""
        print(f"No SageMaker endpoints found{qualifier} in {args.region}.")
        return 0

    # Column widths
    name_w = max(len(ep["EndpointName"]) for ep in endpoints)
    name_w = max(name_w, len("ENDPOINT NAME"))
    status_w = max(len(ep["EndpointStatus"]) for ep in endpoints)
    status_w = max(status_w, len("STATUS"))

    header = (
        f"{'ENDPOINT NAME':<{name_w}}  "
        f"{'STATUS':<{status_w}}  "
        f"{'CREATED':<20}  "
        f"LAST MODIFIED"
    )
    sep = "-" * len(header)

    print(f"\nSageMaker Endpoints  [{args.region}]")
    print(sep)
    print(header)
    print(sep)

    for ep in sorted(endpoints, key=lambda e: e["EndpointName"]):
        name = ep["EndpointName"]
        status = ep["EndpointStatus"]
        created = ep["CreationTime"].strftime("%Y-%m-%d %H:%M:%S")
        modified = ep["LastModifiedTime"].strftime("%Y-%m-%d %H:%M:%S")
        status_str = _colour_status(status, use_colour)
        # Pad without ANSI codes affecting alignment
        pad = status_w - len(status)
        print(f"{name:<{name_w}}  {status_str}{' ' * pad}  {created:<20}  {modified}")

    print(sep)
    qualifier = f" (filtered: {status_filter})" if status_filter else ""
    print(f"{len(endpoints)} endpoint(s){qualifier}\n")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stream audio to multiple simultaneous Deepgram Flux connections "
            "on Amazon SageMaker for stress testing."
        )
    )
    subparsers = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")

    # -- list-endpoints subcommand --------------------------------------------
    list_parser = subparsers.add_parser(
        "list-endpoints",
        help="List available SageMaker endpoints",
        description="List SageMaker endpoints in the target region with their status.",
    )
    list_parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION})",
    )
    list_parser.add_argument(
        "--status",
        default=None,
        metavar="STATUS",
        choices=[s.lower() for s in _STATUS_DISPLAY],
        help=(
            "Filter by endpoint status. "
            "Choices: " + ", ".join(s.lower() for s in _STATUS_DISPLAY)
        ),
    )
    list_parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: WARNING)",
    )

    # -- file subcommand ------------------------------------------------------
    file_parser = subparsers.add_parser(
        "file",
        help="Stream a WAV audio file at real-time pace",
        description=(
            "Stream a 16-bit PCM WAV file in real-time to multiple simultaneous "
            "Flux connections for repeatable load testing."
        ),
    )
    _add_common_args(file_parser)
    file_parser.add_argument(
        "--file",
        required=True,
        metavar="WAV_FILE",
        help="Path to a 16-bit PCM WAV file to stream",
    )
    file_parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the WAV file continuously until --duration is reached or Ctrl+C",
    )

    # -- microphone subcommand ------------------------------------------------
    mic_parser = subparsers.add_parser(
        "microphone",
        help="Capture live microphone input and stream it in real-time",
        description=(
            "Capture live audio from a microphone and stream it to multiple "
            "simultaneous Flux connections for real-time stress testing."
        ),
    )
    _add_common_args(mic_parser)
    mic_parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_MIC_SAMPLE_RATE,
        help=f"Microphone sample rate in Hz (default: {DEFAULT_MIC_SAMPLE_RATE})",
    )
    mic_parser.add_argument(
        "--device",
        type=int,
        default=None,
        metavar="INDEX",
        help="PyAudio input device index (default: system default). Use --list-devices to enumerate.",
    )
    mic_parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        return 0

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    if args.subcommand == "list-endpoints":
        return run_list_endpoints(args)

    err = _validate_common_args(args)
    if err:
        print(f"ERROR: {err}")
        return 1

    if args.subcommand == "file":
        return await run_file(args)
    else:
        return await run_microphone(args)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
