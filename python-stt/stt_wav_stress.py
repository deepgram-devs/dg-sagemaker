#!/usr/bin/env python3
"""
Deepgram SageMaker WAV File STT Client - Multiple Connections

Streams or batch-transcribes a WAV audio file using a Deepgram model
deployed on Amazon SageMaker.

Sub-commands
------------
stream  – Bidirectional streaming via the /v1/listen WebSocket-style
          HTTP/2 endpoint.  Audio is paced to the WAV file's sample rate
          to simulate live input.  Supports multiple simultaneous connections
          for load testing.

batch   – Synchronous HTTP via the SageMaker InvokeEndpoint API and the
          Deepgram pre-recorded /v1/listen REST endpoint.  The entire WAV
          file is posted in a single request.  Supports configurable
          concurrency and a total request count for throughput testing.
"""

import asyncio
import argparse
import concurrent.futures
import json
import logging
import os
import signal
import statistics
import sys
import threading
import time
import wave
from urllib.parse import quote
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from aws_sdk_sagemaker_runtime_http2.client import SageMakerRuntimeHTTP2Client
from aws_sdk_sagemaker_runtime_http2.config import Config, HTTPAuthSchemeResolver
from aws_sdk_sagemaker_runtime_http2.models import (
    InvokeEndpointWithBidirectionalStreamInput,
    RequestStreamEventPayloadPart,
    RequestPayloadPart
)
from smithy_aws_core.identity import EnvironmentCredentialsResolver
from smithy_aws_core.auth.sigv4 import SigV4AuthScheme

# Configuration constants
DEFAULT_REGION = "us-east-2"
CHUNK_SIZE = 8192  # Bytes per audio chunk (stream mode)

# SageMaker InvokeEndpoint body limit (bytes)
INVOKE_ENDPOINT_MAX_BYTES = 6_291_456  # 6 MB

# Deepgram supported redaction entity types
# https://developers.deepgram.com/docs/supported-entity-types
REDACT_ENTITIES = [
    # Broad group redactions
    "pii", "phi", "pci",
    # PII entities
    "account_number", "age", "bank_account", "cardinal", "credit_card",
    "credit_card_expiration", "cvv", "date", "date_interval", "dob",
    "driver_license", "email_address", "event", "filename", "gender_sexuality",
    "healthcare_number", "ip_address", "location", "location_address",
    "location_city", "location_coordinate", "location_country", "location_state",
    "location_zip", "money", "name", "name_given", "name_family",
    "name_medical_professional", "numerical_pii", "occupation", "ordinal",
    "origin", "passport_number", "password", "percent", "phone_number",
    "physical_attribute", "ssn", "time", "url", "username", "vehicle_id",
    # PHI entities
    "condition", "drug", "injury", "blood_type", "medical_process", "statistics",
    # Other entities
    "language", "marital_status", "organization", "political_affiliation",
    "religion", "routing_number", "zodiac_sign",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Streaming classes
# ---------------------------------------------------------------------------

class DeepgramSageMakerConnection:
    """
    Represents a single bidirectional streaming connection to SageMaker.

    Each connection handles its own stream and response processing.
    """

    def __init__(self, connection_id, client, endpoint_name):
        self.connection_id = connection_id
        self.client = client
        self.endpoint_name = endpoint_name
        self.stream = None
        self.output_stream = None
        self.is_active = False
        self.response_task = None
        self.chunk_count = 0

    async def start_session(self, sample_rate, model="nova-3", language="en", **kwargs):
        """
        Start a bidirectional streaming session with Deepgram on SageMaker.

        Args:
            sample_rate: Sample rate of the audio being sent.
            model: Deepgram model (default: nova-3).
            language: Language code (default: en).
            **kwargs: Additional Deepgram query parameters.
        """
        keywords_list = kwargs.pop('keywords', [])
        keyterms_list = kwargs.pop('keyterms', [])
        redact_list = kwargs.pop('redact_entities', [])

        query_params = {
            "model": model,
            "language": language,
            "encoding": "linear16",
            "sample_rate": sample_rate,
        }
        query_params.update(kwargs)

        query_string = "&".join(f"{k}={v}" for k, v in query_params.items())

        if keywords_list:
            keywords_params = "&".join(f"keywords={quote(kw)}" for kw in keywords_list)
            query_string = f"{query_string}&{keywords_params}"

        if keyterms_list:
            keyterms_params = "&".join(f"keyterm={quote(kt)}" for kt in keyterms_list)
            query_string = f"{query_string}&{keyterms_params}"

        if redact_list:
            redact_params = "&".join(f"redact={quote(r)}" for r in redact_list)
            query_string = f"{query_string}&{redact_params}"

        logger.debug(f"[Connection {self.connection_id}] Starting session: {query_string}")

        stream_input = InvokeEndpointWithBidirectionalStreamInput(
            endpoint_name=self.endpoint_name,
            model_invocation_path="v1/listen",
            model_query_string=query_string
        )

        self.stream = await self.client.invoke_endpoint_with_bidirectional_stream(stream_input)
        self.is_active = True

        output = await self.stream.await_output()
        self.output_stream = output[1]

        self.response_task = asyncio.create_task(self._process_responses())
        await asyncio.sleep(0.1)

        logger.info(f"[Connection {self.connection_id}] Session started")

    async def send_audio_chunk(self, audio_bytes):
        """Send a chunk of audio data to the stream."""
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

    async def _process_responses(self):
        """Process streaming responses from Deepgram."""
        try:
            logger.debug(f"[Connection {self.connection_id}] Response processor started")

            while self.is_active:
                result = await self.output_stream.receive()
                if result is None:
                    logger.debug(f"[Connection {self.connection_id}] Stream closed by server")
                    break
                if result.value and result.value.bytes_:
                    self._handle_response(result.value.bytes_.decode('utf-8'))

            logger.debug(f"[Connection {self.connection_id}] Draining remaining responses...")
            drain_count = 0
            while drain_count < 10:
                try:
                    result = await asyncio.wait_for(self.output_stream.receive(), timeout=0.5)
                    if result is None:
                        break
                    if result.value and result.value.bytes_:
                        drain_count += 1
                        self._handle_response(result.value.bytes_.decode('utf-8'))
                except asyncio.TimeoutError:
                    break

        except Exception as e:
            logger.error(
                f"[Connection {self.connection_id}] Error in response processor: {e}",
                exc_info=True,
            )

    def _handle_response(self, raw: str):
        """Parse and print a streaming transcript response."""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"[Connection {self.connection_id}] Non-JSON response: {raw!r}")
            return

        if 'channel' not in parsed:
            return

        alternatives = parsed.get('channel', {}).get('alternatives', [])
        if not alternatives or not alternatives[0].get('transcript'):
            return

        transcript = alternatives[0]['transcript']
        if not transcript.strip():
            return

        confidence = alternatives[0].get('confidence', 0)
        is_final = parsed.get('is_final', False)
        speech_final = parsed.get('speech_final', False)

        if is_final and speech_final:
            print(f"[Conn {self.connection_id}] ✓ {transcript} ({confidence:.1%})")
        else:
            print(f"[Conn {self.connection_id}]   {transcript} [interim]")

    async def end_session(self):
        """Close the streaming session."""
        if not self.is_active:
            return

        logger.debug(f"[Connection {self.connection_id}] Ending session")
        self.is_active = False

        try:
            await self.stream.input_stream.close()
        except Exception as e:
            logger.error(f"[Connection {self.connection_id}] Error closing input stream: {e}")

        if self.response_task and not self.response_task.done():
            try:
                await asyncio.wait_for(self.response_task, timeout=15.0)
            except asyncio.TimeoutError:
                logger.warning(
                    f"[Connection {self.connection_id}] Timeout waiting for final responses"
                )
                self.response_task.cancel()
            except asyncio.CancelledError:
                pass

        logger.info(
            f"[Connection {self.connection_id}] Session ended "
            f"(sent {self.chunk_count} chunks)"
        )


class MultiConnectionWAVClient:
    """
    Streams a WAV file in real-time to multiple simultaneous Deepgram
    connections on AWS SageMaker.

    Audio chunks are paced according to the WAV file's sample rate so that
    the stream arrives at the endpoint at the same rate as live audio would.
    The file loops if --loop is specified, otherwise it plays once.
    """

    def __init__(self, endpoint_name, wav_path, region=DEFAULT_REGION, num_connections=1):
        self.endpoint_name = endpoint_name
        self.wav_path = wav_path
        self.region = region
        self.num_connections = num_connections
        self.bidi_endpoint = f"https://runtime.sagemaker.{region}.amazonaws.com:8443"
        self.client = None
        self.connections = []
        self.is_active = False

        self.sample_rate = None
        self.channels = None
        self.sample_width = None
        self.duration_seconds = None

    def _open_wav(self):
        """Read WAV header and validate, returning a wave.Wave_read object."""
        try:
            wf = wave.open(self.wav_path, 'rb')
        except FileNotFoundError:
            logger.error(f"WAV file not found: {self.wav_path}")
            raise
        except wave.Error as e:
            logger.error(f"Invalid WAV file '{self.wav_path}': {e}")
            raise

        self.sample_rate = wf.getframerate()
        self.channels = wf.getnchannels()
        self.sample_width = wf.getsampwidth()
        self.duration_seconds = wf.getnframes() / self.sample_rate

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

    def _initialize_client(self):
        """Initialize the SageMaker Runtime HTTP/2 client with AWS credentials."""
        try:
            session = boto3.Session(region_name=self.region)
            credentials = session.get_credentials()
            if credentials is None:
                raise NoCredentialsError()

            frozen = credentials.get_frozen_credentials()
            os.environ['AWS_ACCESS_KEY_ID'] = frozen.access_key
            os.environ['AWS_SECRET_ACCESS_KEY'] = frozen.secret_key
            if frozen.token:
                os.environ['AWS_SESSION_TOKEN'] = frozen.token

            caller_identity = session.client('sts').get_caller_identity()
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
            auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="sagemaker")}
        )
        self.client = SageMakerRuntimeHTTP2Client(config=config)
        logger.info("SageMaker streaming client initialized")

    async def initialize_connections(self, model="nova-3", language="en", **kwargs):
        """Start all bidirectional streaming connections in parallel."""
        if not self.client:
            self._initialize_client()

        logger.info(f"Initializing {self.num_connections} connection(s)...")
        for i in range(self.num_connections):
            conn = DeepgramSageMakerConnection(i + 1, self.client, self.endpoint_name)
            self.connections.append(conn)

        await asyncio.gather(*[
            conn.start_session(self.sample_rate, model=model, language=language, **kwargs)
            for conn in self.connections
        ])

        self.is_active = True
        logger.info(f"All {self.num_connections} connection(s) started")

    async def stream_wav_audio(self, loop=False):
        """
        Read the WAV file and broadcast audio chunks to all connections at
        real-time speed, paced by the file's sample rate.
        """
        bytes_per_frame = self.sample_width * self.channels
        frames_per_chunk = CHUNK_SIZE // bytes_per_frame
        chunk_duration = frames_per_chunk / self.sample_rate

        play_count = 0
        total_chunks = 0

        while self.is_active:
            wf = self._open_wav()
            play_count += 1
            logger.info(f"Streaming WAV file (pass {play_count})...")
            chunk_start = time.monotonic()

            while self.is_active:
                raw = wf.readframes(frames_per_chunk)
                if not raw:
                    break

                active = [c for c in self.connections if c.is_active]
                if not active:
                    logger.warning("All connections have failed")
                    self.is_active = False
                    break

                await asyncio.gather(*[c.send_audio_chunk(raw) for c in active])
                total_chunks += 1

                elapsed = time.monotonic() - chunk_start
                sleep_time = total_chunks * chunk_duration - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            wf.close()
            if not loop:
                break

        logger.info(f"Finished streaming: {play_count} pass(es), {total_chunks} chunks total")

    async def end_all_sessions(self):
        """Close all streaming sessions."""
        if not self.is_active and not self.connections:
            return
        self.is_active = False
        logger.info(f"Closing {len(self.connections)} connection(s)...")
        await asyncio.gather(*[c.end_session() for c in self.connections])
        logger.info("All sessions ended")


# ---------------------------------------------------------------------------
# Batch (pre-recorded) class
# ---------------------------------------------------------------------------

class BatchSTTClient:
    """
    Sends a WAV audio file to a Deepgram STT SageMaker endpoint using the
    synchronous InvokeEndpoint API (pre-recorded / batch mode).

    Deepgram parameters are passed via the
    X-Amzn-SageMaker-Custom-Attributes header, which is forwarded verbatim
    to the container and interpreted as a URL query string.

    Supports concurrent requests with configurable parallelism for throughput
    and latency stress testing.

    Note: SageMaker InvokeEndpoint has a 6 MB request body limit. WAV files
    larger than this cannot be submitted in batch mode; use stream mode or
    split the audio into shorter segments.
    """

    def __init__(self, endpoint_name: str, wav_path: str, region: str = DEFAULT_REGION):
        self.endpoint_name = endpoint_name
        self.wav_path = wav_path
        self.region = region
        self._session: boto3.Session | None = None
        self._thread_local = threading.local()

        self.sample_rate: int | None = None
        self.channels: int | None = None
        self.duration_seconds: float | None = None

    def _initialize_client(self):
        """Resolve AWS credentials and verify identity. Stores the session for per-thread client creation."""
        try:
            session = boto3.Session(region_name=self.region)
            credentials = session.get_credentials()
            if credentials is None:
                raise NoCredentialsError()
            credentials.get_frozen_credentials()  # force resolution

            caller_identity = session.client('sts').get_caller_identity()
            logger.debug(f"Authenticated as: {caller_identity.get('Arn', 'Unknown')}")

            self._session = session
            logger.info("AWS credentials resolved")

        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error("AWS credentials not found. Configure via one of:")
            logger.error("  1. aws configure")
            logger.error("  2. Environment variables: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY")
            logger.error("  3. ~/.aws/credentials")
            logger.error("  4. IAM role (when running on AWS infrastructure)")
            raise RuntimeError("AWS credentials not available") from e

    def _get_thread_client(self):
        """Return a per-thread boto3 sagemaker-runtime client, creating one if needed."""
        if not hasattr(self._thread_local, 'sm_client'):
            self._thread_local.sm_client = self._session.client('sagemaker-runtime')
            logger.debug(f"Created sagemaker-runtime client for thread {threading.current_thread().name}")
        return self._thread_local.sm_client

    def load_wav(self) -> bytes:
        """
        Read and validate the WAV file, returning its raw bytes.

        Populates sample_rate, channels, and duration_seconds as a side effect.
        Raises ValueError for non-16-bit PCM files or files exceeding the
        SageMaker InvokeEndpoint 6 MB body limit.
        """
        try:
            with wave.open(self.wav_path, 'rb') as wf:
                sample_width = wf.getsampwidth()
                if sample_width != 2:
                    raise ValueError(
                        f"WAV file must be 16-bit PCM (sample width 2 bytes). "
                        f"Got {sample_width * 8}-bit audio. "
                        "Convert with: ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 output.wav"
                    )
                self.sample_rate = wf.getframerate()
                self.channels = wf.getnchannels()
                self.duration_seconds = wf.getnframes() / self.sample_rate
        except FileNotFoundError:
            logger.error(f"WAV file not found: {self.wav_path}")
            raise
        except wave.Error as e:
            logger.error(f"Invalid WAV file '{self.wav_path}': {e}")
            raise

        with open(self.wav_path, 'rb') as f:
            audio_bytes = f.read()

        if len(audio_bytes) > INVOKE_ENDPOINT_MAX_BYTES:
            mb = len(audio_bytes) / 1_048_576
            raise ValueError(
                f"WAV file is {mb:.1f} MB, which exceeds the SageMaker InvokeEndpoint "
                f"6 MB body limit. Use stream mode for longer audio, or split the file "
                "into shorter segments with: "
                "ffmpeg -i input.wav -f segment -segment_time 60 segment_%03d.wav"
            )

        logger.info(
            f"WAV: {self.wav_path} | {self.sample_rate} Hz | "
            f"{self.channels}ch | {self.duration_seconds:.2f}s | "
            f"{len(audio_bytes) / 1024:.1f} KB"
        )
        return audio_bytes

    def _invoke_once(
        self,
        request_id: int,
        audio_bytes: bytes,
        custom_attributes: str,
    ) -> tuple[int, float, dict | None, Exception | None]:
        """
        Make a single InvokeEndpoint call on the calling thread.

        Uses a per-thread boto3 client so concurrent calls don't share connections.
        Returns (request_id, elapsed_seconds, parsed_response_or_None, error_or_None).
        """
        sm_client = self._get_thread_client()
        start = time.monotonic()
        try:
            response = sm_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                Body=audio_bytes,
                ContentType='audio/wav',
                Accept='application/json',
                CustomAttributes=custom_attributes,
            )
            elapsed = time.monotonic() - start
            body_bytes = response['Body'].read()
            result = json.loads(body_bytes)
            return request_id, elapsed, result, None

        except ClientError as e:
            elapsed = time.monotonic() - start
            code = e.response['Error']['Code']
            msg = e.response['Error']['Message']
            err = RuntimeError(
                f"SageMaker InvokeEndpoint failed [{code}]: {msg}. "
                "Check the endpoint name, region, and that the endpoint is InService."
            )
            return request_id, elapsed, None, err

        except Exception as e:
            elapsed = time.monotonic() - start
            return request_id, elapsed, None, e

    def run(
        self,
        custom_attributes: str,
        num_requests: int = 1,
        concurrency: int = 1,
    ) -> list[tuple[int, float, dict | None, Exception | None]]:
        """
        Run num_requests InvokeEndpoint calls using a ThreadPoolExecutor with
        up to concurrency threads in parallel.  Each thread maintains its own
        boto3 sagemaker-runtime client to avoid connection contention.

        Returns a list of (request_id, elapsed, result, error) tuples in
        completion order.
        """
        if self._session is None:
            self._initialize_client()

        audio_bytes = self.load_wav()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix="batch-invoke",
        ) as executor:
            futures = {
                executor.submit(self._invoke_once, i + 1, audio_bytes, custom_attributes): i + 1
                for i in range(num_requests)
            }
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_redact(value: str) -> list[str]:
    """
    Parse a comma-separated --redact string into a validated list of entity types.

    Raises SystemExit with an error message for any unrecognised entity.
    """
    if not value:
        return []
    entities = []
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        if item not in REDACT_ENTITIES:
            print(
                f"ERROR: '{item}' is not a valid redact entity. "
                f"Supported values: {', '.join(REDACT_ENTITIES)}"
            )
            raise SystemExit(1)
        entities.append(item)
    return entities


def _build_batch_query_string(args) -> str:
    """
    Build a Deepgram invocation path + query string for pre-recorded mode.

    The string is passed as X-Amzn-SageMaker-Custom-Attributes and forwarded
    to the Deepgram container.  It must include the API path so the container
    can route the request to the correct Deepgram endpoint, matching how the
    streaming mode passes model_invocation_path="v1/listen".
    """
    params: list[str] = [
        f"model={args.model}",
        f"language={args.language}",
        f"punctuate={args.punctuate}",
        f"diarize={args.diarize}",
    ]
    if args.redact:
        params.extend(f"redact={quote(r)}" for r in args.redact)
    if args.keyterms:
        params.extend(f"keyterm={quote(kt)}" for kt in args.keyterms)
    return "v1/listen?" + "&".join(params)


def _extract_transcript(result: dict) -> tuple[str, float]:
    """
    Pull the primary transcript and confidence from a Deepgram pre-recorded response.

    Returns ("", 0.0) if the response structure is unexpected.
    """
    try:
        alt = result['results']['channels'][0]['alternatives'][0]
        return alt.get('transcript', ''), alt.get('confidence', 0.0)
    except (KeyError, IndexError):
        return '', 0.0


def _add_common_args(parser: argparse.ArgumentParser):
    """Add arguments shared by both stream and batch subcommands."""
    parser.add_argument("endpoint_name", help="SageMaker endpoint name")
    parser.add_argument(
        "--file",
        required=True,
        metavar="WAV_FILE",
        help="Path to a 16-bit PCM WAV file",
    )
    parser.add_argument(
        "--model",
        default="nova-3",
        help="Deepgram model (default: nova-3)",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code (default: en)",
    )
    parser.add_argument(
        "--punctuate",
        default="true",
        choices=["true", "false"],
        help="Enable punctuation (default: true)",
    )
    parser.add_argument(
        "--diarize",
        default="false",
        choices=["true", "false"],
        help="Enable speaker diarization (default: false)",
    )
    parser.add_argument(
        "--redact",
        default="",
        metavar="ENTITY,ENTITY,...",
        help=(
            "Comma-separated list of entity types to redact from transcripts. "
            f"Supported values: {', '.join(REDACT_ENTITIES)}"
        ),
    )
    parser.add_argument(
        "--keyterms",
        default="",
        metavar="TERM,TERM,...",
        help=(
            "Comma-separated list of keyterms to boost recognition for "
            "(e.g., 'Deepgram,SageMaker'). Each term is sent as keyterm=<value>."
        ),
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO)",
    )


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

async def run_stream(args) -> int:
    keywords_list = []
    if args.keywords:
        for kw in args.keywords.split(','):
            kw = kw.strip()
            if kw:
                keywords_list.append(kw)

    keyterms_list = []
    if args.keyterms:
        for kt in args.keyterms.split(','):
            kt = kt.strip()
            if kt:
                keyterms_list.append(kt)

    redact_list = _parse_redact(args.redact)

    if args.connections < 1:
        print("ERROR: --connections must be a positive integer (minimum 1)")
        return 1

    client = MultiConnectionWAVClient(
        endpoint_name=args.endpoint_name,
        wav_path=args.file,
        region=args.region,
        num_connections=args.connections,
    )

    try:
        wf = client._open_wav()
        wf.close()
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    limit_str = (
        f"{args.duration}s" if args.duration
        else ("until file ends (looping)" if args.loop else "until file ends or Ctrl+C")
    )

    print("=" * 60)
    print("Deepgram SageMaker WAV Streaming Client")
    print("=" * 60)
    print(f"Endpoint:    {args.endpoint_name}")
    print(f"WAV File:    {args.file}")
    print(f"Duration:    {client.duration_seconds:.2f}s")
    print(f"Sample Rate: {client.sample_rate} Hz")
    print(f"Channels:    {client.channels}")
    print(f"Connections: {args.connections}")
    print(f"Model:       {args.model}")
    print(f"Language:    {args.language}")
    print(f"Region:      {args.region}")
    print(f"Loop:        {'yes' if args.loop else 'no'}")
    print(f"Limit:       {limit_str}")
    if redact_list:
        print(f"Redact:      {', '.join(redact_list)}")
    if keyterms_list:
        print(f"Keyterms:    {', '.join(keyterms_list)}")
    print("=" * 60)

    def signal_handler(sig, frame):
        print("\n\nReceived interrupt signal, stopping...")
        client.is_active = False

    signal.signal(signal.SIGINT, signal_handler)

    try:
        await client.initialize_connections(
            model=args.model,
            language=args.language,
            diarize=args.diarize,
            punctuate=args.punctuate,
            interim_results=args.interim_results,
            keywords=keywords_list,
            keyterms=keyterms_list,
            redact_entities=redact_list,
        )

        print("\n" + "=" * 60)
        print(f"LIVE TRANSCRIPTION - {args.connections} Connection(s)")
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

    logger.info("Transcription complete")
    return 0


async def run_batch(args) -> int:
    redact_list = _parse_redact(args.redact)

    if args.concurrency < 1:
        print("ERROR: --concurrency must be a positive integer (minimum 1)")
        return 1
    if args.requests is None:
        args.requests = args.concurrency
    if args.requests < 1:
        print("ERROR: --requests must be a positive integer (minimum 1)")
        return 1
    if args.concurrency > args.requests:
        print("WARNING: --concurrency exceeds --requests; clamping to request count")
        args.concurrency = args.requests

    client = BatchSTTClient(
        endpoint_name=args.endpoint_name,
        wav_path=args.file,
        region=args.region,
    )

    # Load WAV early for validation and metadata
    try:
        client._initialize_client()
        audio_bytes = client.load_wav()
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    args.redact = redact_list
    args.keyterms = [kt.strip() for kt in args.keyterms.split(',') if kt.strip()]
    custom_attributes = _build_batch_query_string(args)

    print("=" * 60)
    print("Deepgram SageMaker WAV Batch Client (Pre-Recorded)")
    print("=" * 60)
    print(f"Endpoint:     {args.endpoint_name}")
    print(f"WAV File:     {args.file}")
    print(f"File Size:    {len(audio_bytes) / 1024:.1f} KB")
    print(f"Duration:     {client.duration_seconds:.2f}s")
    print(f"Sample Rate:  {client.sample_rate} Hz")
    print(f"Channels:     {client.channels}")
    print(f"Model:        {args.model}")
    print(f"Language:     {args.language}")
    print(f"Region:       {args.region}")
    print(f"Requests:     {args.requests}")
    print(f"Concurrency:  {args.concurrency}")
    if redact_list:
        print(f"Redact:       {', '.join(redact_list)}")
    if args.keyterms:
        print(f"Keyterms:     {args.keyterms}")
    print(f"Path+Params:  {custom_attributes}")
    print("=" * 60)

    print(f"\nRunning {args.requests} request(s) "
          f"({args.concurrency} concurrent)...\n")

    wall_start = time.monotonic()

    results = await asyncio.to_thread(
        client.run,
        custom_attributes,
        args.requests,
        args.concurrency,
    )

    wall_elapsed = time.monotonic() - wall_start

    # Sort by request_id for consistent output
    results.sort(key=lambda r: r[0])

    successes = 0
    failures = 0
    latencies: list[float] = []

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    for req_id, elapsed, result, error in results:
        if error:
            failures += 1
            print(f"[Request {req_id:>3}] ERROR ({elapsed:.2f}s): {error}")
        else:
            successes += 1
            latencies.append(elapsed)
            transcript, confidence = _extract_transcript(result)
            if transcript.strip():
                print(
                    f"[Request {req_id:>3}] ✓ ({elapsed:.2f}s) "
                    f"({confidence:.1%}) {transcript}"
                )
            else:
                print(f"[Request {req_id:>3}] ✓ ({elapsed:.2f}s) (no transcript)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total requests:  {args.requests}")
    print(f"Successful:      {successes}")
    print(f"Failed:          {failures}")

    if latencies:
        latencies_sorted = sorted(latencies)
        p95_idx = max(0, int(len(latencies_sorted) * 0.95) - 1)
        print(f"Min latency:     {min(latencies):.2f}s")
        print(f"Avg latency:     {statistics.mean(latencies):.2f}s")
        print(f"P95 latency:     {latencies_sorted[p95_idx]:.2f}s")
        print(f"Max latency:     {max(latencies):.2f}s")
        if len(latencies) > 1:
            print(f"Std dev:         {statistics.stdev(latencies):.2f}s")

    print(f"Total wall time: {wall_elapsed:.2f}s")
    if args.requests > 0 and wall_elapsed > 0:
        print(f"Throughput:      {args.requests / wall_elapsed:.2f} req/s")
    print("=" * 60)

    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stream or batch-transcribe a WAV file using a Deepgram model "
            "deployed on Amazon SageMaker."
        )
    )
    subparsers = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")

    # -- stream subcommand ----------------------------------------------------
    stream_parser = subparsers.add_parser(
        "stream",
        help="Bidirectional streaming transcription (real-time, /v1/listen WebSocket)",
        description=(
            "Stream a WAV file in real-time to one or more simultaneous Deepgram "
            "connections on SageMaker using bidirectional HTTP/2 streaming."
        ),
    )
    _add_common_args(stream_parser)
    stream_parser.add_argument(
        "--connections",
        type=int,
        default=1,
        help="Number of simultaneous streaming connections (default: 1)",
    )
    stream_parser.add_argument(
        "--keywords",
        default="",
        help="Comma-delimited keywords in format 'keyword:intensity' (e.g., 'hello:5,world:10')",
    )
    stream_parser.add_argument(
        "--interim-results",
        default="true",
        choices=["true", "false"],
        help="Enable interim (partial) results (default: true)",
    )
    stream_parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the WAV file continuously until --duration is reached or Ctrl+C",
    )
    stream_parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop automatically after this many seconds (default: run until file ends or Ctrl+C)",
    )

    # -- batch subcommand -----------------------------------------------------
    batch_parser = subparsers.add_parser(
        "batch",
        help="Synchronous pre-recorded transcription (HTTP POST, SageMaker InvokeEndpoint)",
        description=(
            "Submit a WAV file as a single HTTP POST using the SageMaker InvokeEndpoint "
            "API and the Deepgram pre-recorded /v1/listen REST endpoint. "
            f"File size is limited to {INVOKE_ENDPOINT_MAX_BYTES // 1_048_576} MB by "
            "SageMaker. Use --requests and --concurrency to stress-test throughput."
        ),
    )
    _add_common_args(batch_parser)
    batch_parser.add_argument(
        "--requests",
        type=int,
        default=None,
        help="Total number of InvokeEndpoint requests to send (default: same as --concurrency)",
    )
    batch_parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of requests to run in parallel (default: 1)",
    )

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        return 0

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True,
    )

    if args.subcommand == "stream":
        return await run_stream(args)
    else:
        return await run_batch(args)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
