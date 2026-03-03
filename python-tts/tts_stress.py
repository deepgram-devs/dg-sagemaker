#!/usr/bin/env python3
"""
Deepgram SageMaker Text-to-Speech Streaming Client - Multiple Connections

This client generates text and streams it to multiple simultaneous bidirectional
connections to a Deepgram TTS model deployed on SageMaker for real-time synthesis.
Useful for load testing and stress testing endpoints.

Audio from a single selected connection is played back to local system speakers,
while other connections receive audio but discard it.
"""

import asyncio
import argparse
import json
import logging
import os
import signal
import sys
import time
from queue import Queue
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from aws_sdk_sagemaker_runtime_http2.client import SageMakerRuntimeHTTP2Client
from aws_sdk_sagemaker_runtime_http2.config import Config, HTTPAuthSchemeResolver
from aws_sdk_sagemaker_runtime_http2.models import (
    InvokeEndpointWithBidirectionalStreamInput,
    RequestStreamEventPayloadPart,
    RequestPayloadPart
)
from smithy_aws_core.identity import EnvironmentCredentialsResolver
from smithy_aws_core.auth.sigv4 import SigV4AuthScheme

try:
    import pyaudio
except ImportError:
    print("ERROR: pyaudio is required for audio playback")
    print("Install it with: pip install pyaudio")
    print("On macOS, you may need: brew install portaudio && pip install pyaudio")
    sys.exit(1)

# Configuration constants
DEFAULT_REGION = "us-east-2"
AUDIO_FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1  # Mono audio
SAMPLE_RATE = 24000  # 24kHz sample rate (typical for TTS)

logger = logging.getLogger(__name__)


class DeepgramSageMakerTTSConnection:
    """
    Represents a single bidirectional streaming connection to SageMaker for TTS.

    Each connection handles its own stream and audio response processing.
    """

    def __init__(self, connection_id, client, endpoint_name, should_playback=False, boto_session=None):
        self.connection_id = connection_id
        self.client = client
        self.endpoint_name = endpoint_name
        self.should_playback = should_playback
        self.boto_session = boto_session
        self.stream = None
        self.output_stream = None
        self.is_active = False
        self.close_sent = False
        self.response_task = None
        self.audio_buffer = Queue()
        self.bytes_received = 0
        self.phrase_count = 0
        # Set initially so the first phrase can be sent without waiting.
        self._flushed_event = asyncio.Event()
        self._flushed_event.set()
        self._last_flush_time = 0.0

    async def start_session(self, voice="aura-2-thalia-en", **kwargs):
        """
        Start a bidirectional streaming session with Deepgram TTS on SageMaker

        Args:
            voice: Deepgram TTS voice to use (default: aura-2-thalia-en)
            **kwargs: Additional Deepgram query parameters
        """
        # Build query string for Deepgram parameters
        query_params = {
            "model": voice,
            "encoding": "linear16",
            "sample_rate": SAMPLE_RATE,
        }
        query_params.update(kwargs)

        # Convert dict to query string
        query_string = "&".join(f"{k}={v}" for k, v in query_params.items())

        logger.debug(f"[Connection {self.connection_id}] Starting session with endpoint: {self.endpoint_name}")
        logger.debug(f"[Connection {self.connection_id}] Deepgram parameters: {query_string}")

        # Create the bidirectional stream
        stream_input = InvokeEndpointWithBidirectionalStreamInput(
            endpoint_name=self.endpoint_name,
            model_invocation_path="v1/speak",
            model_query_string=query_string
        )

        logger.debug(f"[Connection {self.connection_id}] Invoking endpoint with bidirectional stream")
        self.stream = await self.client.invoke_endpoint_with_bidirectional_stream(stream_input)
        self.is_active = True

        logger.debug(f"[Connection {self.connection_id}] Stream created, connecting to output stream")
        # Get output stream immediately before starting background task
        output = await self.stream.await_output()
        self.output_stream = output[1]
        logger.debug(f"[Connection {self.connection_id}] Connected to output stream")

        # Start processing responses in the background
        logger.debug(f"[Connection {self.connection_id}] Starting response processor")
        self.response_task = asyncio.create_task(self._process_audio_responses())

        # Give the response processor a moment to start
        await asyncio.sleep(0.1)

        logger.info(f"[Connection {self.connection_id}] Session started successfully")

    async def send_text(self, text):
        """
        Send text to the TTS stream in Deepgram API format

        Args:
            text: Text to convert to speech

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.is_active or self.close_sent:
            return False

        try:
            # Format message according to Deepgram TTS API specification
            message = {
                "type": "Speak",
                "text": text
            }
            message_bytes = json.dumps(message).encode('utf-8')

            # Create payload with DataType specified for SageMaker wrapper
            payload = RequestPayloadPart(bytes_=message_bytes, data_type="UTF8")
            event = RequestStreamEventPayloadPart(value=payload)
            await self.stream.input_stream.send(event)
            self.phrase_count += 1
            logger.info(f"[Connection {self.connection_id}] Sent text chunk {self.phrase_count} ({len(message_bytes)} bytes): {text!r}")
            return True
        except Exception as e:
            logger.error(f"[Connection {self.connection_id}] Error sending text: {e}")
            self.is_active = False
            return False

    async def send_close_message(self):
        """
        Send a Close message to signal end of transmission according to Deepgram API spec

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.is_active:
            return False

        try:
            # Format Close message according to Deepgram TTS API specification
            message = {"type": "Close"}
            message_bytes = json.dumps(message).encode('utf-8')

            # Create payload with DataType specified for SageMaker wrapper
            payload = RequestPayloadPart(bytes_=message_bytes, data_type="UTF8")
            event = RequestStreamEventPayloadPart(value=payload)
            await self.stream.input_stream.send(event)
            self.close_sent = True
            logger.info(f"[Connection {self.connection_id}] Sent Close message")
            return True
        except Exception as e:
            logger.error(f"[Connection {self.connection_id}] Error sending Close message: {e}")
            return False

    async def send_flush(self):
        """
        Send a Flush message to force Deepgram to synthesize buffered text immediately.

        Clears the flushed event before sending so that wait_for_flushed() will
        block until the server acknowledges this flush with a Flushed response.

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.is_active or self.close_sent:
            return False

        # Deepgram enforces a maximum of 20 Flush messages per 60-second window
        # per connection. Sleep the remaining time if we're within that window so
        # text sent since the last flush continues to accumulate in the buffer.
        MIN_FLUSH_INTERVAL = 3.0
        now = time.monotonic()
        wait = (self._last_flush_time + MIN_FLUSH_INTERVAL) - now
        if wait > 0:
            logger.debug(f"[Connection {self.connection_id}] Flush rate limit: waiting {wait:.2f}s")
            await asyncio.sleep(wait)

        if not self.is_active or self.close_sent:
            self._flushed_event.set()
            return False

        self._flushed_event.clear()
        try:
            message = {"type": "Flush"}
            message_bytes = json.dumps(message).encode('utf-8')
            payload = RequestPayloadPart(bytes_=message_bytes, data_type="UTF8")
            event = RequestStreamEventPayloadPart(value=payload)
            await self.stream.input_stream.send(event)
            self._last_flush_time = time.monotonic()
            logger.debug(f"[Connection {self.connection_id}] Sent Flush")
            return True
        except Exception as e:
            logger.error(f"[Connection {self.connection_id}] Error sending Flush: {e}")
            self.is_active = False
            self._flushed_event.set()  # Unblock any waiters
            return False

    async def wait_for_flushed(self, timeout: float = 30.0) -> bool:
        """
        Block until the server acknowledges the last Flush with a Flushed message.

        Returns immediately if the connection is inactive or no flush is in flight.

        Args:
            timeout: Maximum seconds to wait before marking the connection failed.

        Returns:
            True if Flushed was received (or no flush was outstanding), False on timeout.
        """
        if not self.is_active:
            return True
        try:
            await asyncio.wait_for(self._flushed_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.error(
                f"[Connection {self.connection_id}] Timed out waiting for Flushed "
                f"acknowledgment after {timeout}s — marking connection failed"
            )
            self.is_active = False
            return False

    def _log_exception_details(self, exc: Exception):
        """
        Walk the full exception chain and print every piece of structured detail
        that smithy / botocore / AWS SDK exceptions may carry.
        """
        prefix = f"[Connection {self.connection_id}]"

        # Collect the full cause chain without infinite loops
        chain: list[Exception] = []
        seen: set[int] = set()
        current: Exception | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            chain.append(current)
            cause = current.__cause__
            if cause is None and not current.__suppress_context__:
                cause = current.__context__
            current = cause

        print(f"\n--- Error details [Connection {self.connection_id}] ---", file=sys.stderr)

        for depth, err in enumerate(chain):
            indent = "  " * depth
            label = "Exception" if depth == 0 else "Caused by"
            print(f"{indent}{label}: {type(err).__name__}: {err}", file=sys.stderr)
            logger.error(f"{prefix} {label}: {type(err).__name__}: {err}")

            # Smithy ServiceError / AWS SDK typed fields
            for attr in ("code", "fault", "message", "request_id", "error_code"):
                val = getattr(err, attr, None)
                if val and str(val) not in str(err):
                    print(f"{indent}  .{attr} = {val}", file=sys.stderr)
                    logger.error(f"{prefix}{indent}  .{attr} = {val}")

            # HTTP response object (smithy-python style)
            http_resp = getattr(err, "http_response", None)
            if http_resp:
                status = getattr(http_resp, "status", None) or getattr(http_resp, "status_code", None)
                if status:
                    print(f"{indent}  http_status = {status}", file=sys.stderr)
                    logger.error(f"{prefix}{indent}  http_status = {status}")
                body = getattr(http_resp, "body", None)
                if isinstance(body, (bytes, bytearray)):
                    self._print_body(body, indent, prefix)

            # Direct .body attribute (some smithy versions)
            body = getattr(err, "body", None)
            if isinstance(body, (bytes, bytearray)):
                self._print_body(body, indent, prefix)

            # botocore-style .response dict
            response = getattr(err, "response", None)
            if isinstance(response, dict):
                error_info = response.get("Error", {})
                if error_info:
                    code = error_info.get("Code", "")
                    msg = error_info.get("Message", "")
                    print(f"{indent}  Error.Code = {code}", file=sys.stderr)
                    print(f"{indent}  Error.Message = {msg}", file=sys.stderr)
                    logger.error(f"{prefix}{indent}  Error.Code={code}, Error.Message={msg}")

        print("--- End error details ---\n", file=sys.stderr)

    def _print_body(self, body: bytes, indent: str, log_prefix: str):
        """Decode and print a raw HTTP response body, pretty-printing JSON if possible."""
        try:
            parsed = json.loads(body.decode("utf-8"))
            text = json.dumps(parsed, indent=2)
            print(f"{indent}  response_body =\n{text}", file=sys.stderr)
            logger.error(f"{log_prefix}{indent}  response_body = {text}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f"{indent}  response_body (raw) = {body!r}", file=sys.stderr)
            logger.error(f"{log_prefix}{indent}  response_body (raw) = {body!r}")

    async def _process_audio_responses(self):
        """Process streaming responses from Deepgram according to API specification"""
        try:
            logger.debug(f"[Connection {self.connection_id}] Response processor started")

            while self.is_active:
                result = await self.output_stream.receive()

                if result is None:
                    logger.debug(f"[Connection {self.connection_id}] No more responses from server")
                    break

                if result.value and result.value.bytes_:
                    data = result.value.bytes_

                    # Try to parse as JSON control message first
                    is_json_message = False
                    try:
                        message = json.loads(data.decode('utf-8'))
                        is_json_message = True
                        message_type = message.get('type', 'Unknown')

                        if message_type == 'Metadata':
                            uuids = message.get('additional_model_uuids') or []
                            logger.debug(
                                f"[Connection {self.connection_id}] Metadata: "
                                f"request_id={message.get('request_id')} "
                                f"model={message.get('model_name')} "
                                f"version={message.get('model_version')} "
                                f"model_uuid={message.get('model_uuid')}"
                                + (f" additional_uuids={uuids}" if uuids else "")
                            )

                        elif message_type == 'Warning':
                            logger.warning(
                                f"[Connection {self.connection_id}] Warning "
                                f"[{message.get('warn_code')}]: {message.get('warn_msg')}"
                            )

                        elif message_type == 'Flushed':
                            self._flushed_event.set()
                            logger.debug(
                                f"[Connection {self.connection_id}] Flushed "
                                f"(sequence_id={message.get('sequence_id')})"
                            )

                        elif message_type == 'Cleared':
                            logger.debug(
                                f"[Connection {self.connection_id}] Cleared "
                                f"(sequence_id={message.get('sequence_id')})"
                            )

                        elif message_type == 'Error':
                            # Not part of the Deepgram TTS spec; may originate from the
                            # SageMaker inference layer.
                            err_code = message.get('err_code') or message.get('code', 'unknown')
                            description = message.get('description') or message.get('message', 'No description provided')
                            err_msg = message.get('err_msg', '')
                            detail = f" — {err_msg}" if err_msg else ""
                            logger.error(
                                f"[Connection {self.connection_id}] Error "
                                f"[{err_code}]: {description}{detail}"
                            )
                            print(
                                f"ERROR [Connection {self.connection_id}]: [{err_code}] {description}{detail}",
                                file=sys.stderr
                            )

                        else:
                            logger.warning(
                                f"[Connection {self.connection_id}] Unrecognised message type: "
                                f"{message_type!r} — {message}"
                            )

                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # Not JSON - treat as binary audio data
                        is_json_message = False

                    # If not a JSON message, treat as audio data
                    if not is_json_message:
                        self.bytes_received += len(data)

                        if self.should_playback:
                            # Buffer audio for playback
                            self.audio_buffer.put(data)
                            logger.debug(f"[Connection {self.connection_id}] Received {len(data)} bytes of audio (total: {self.bytes_received})")
                        else:
                            logger.debug(f"[Connection {self.connection_id}] Received {len(data)} bytes of audio (discarded)")

            # Continue processing any remaining buffered responses
            logger.debug(f"[Connection {self.connection_id}] Processing any remaining buffered responses...")
            remaining_count = 0
            while remaining_count < 20:
                try:
                    result = await asyncio.wait_for(self.output_stream.receive(), timeout=0.5)
                    if result is None:
                        break

                    if result.value and result.value.bytes_:
                        data = result.value.bytes_

                        # Try to parse as JSON control message
                        is_json_message = False
                        try:
                            message = json.loads(data.decode('utf-8'))
                            is_json_message = True
                            message_type = message.get('type', 'Unknown')

                            if message_type == 'Metadata':
                                uuids = message.get('additional_model_uuids') or []
                                logger.debug(
                                    f"[Connection {self.connection_id}] Metadata: "
                                    f"request_id={message.get('request_id')} "
                                    f"model={message.get('model_name')} "
                                    f"version={message.get('model_version')} "
                                    f"model_uuid={message.get('model_uuid')}"
                                    + (f" additional_uuids={uuids}" if uuids else "")
                                )
                            elif message_type == 'Warning':
                                logger.warning(
                                    f"[Connection {self.connection_id}] Warning "
                                    f"[{message.get('code')}]: {message.get('description')}"
                                )
                            elif message_type in ('Flushed', 'Cleared'):
                                logger.debug(
                                    f"[Connection {self.connection_id}] {message_type} "
                                    f"(sequence_id={message.get('sequence_id')})"
                                )
                            elif message_type == 'Error':
                                err_code = message.get('err_code') or message.get('code', 'unknown')
                                description = message.get('description') or message.get('message', 'No description provided')
                                err_msg = message.get('err_msg', '')
                                detail = f" — {err_msg}" if err_msg else ""
                                logger.error(
                                    f"[Connection {self.connection_id}] Error "
                                    f"[{err_code}]: {description}{detail}"
                                )
                                print(
                                    f"ERROR [Connection {self.connection_id}]: [{err_code}] {description}{detail}",
                                    file=sys.stderr
                                )
                            else:
                                logger.warning(
                                    f"[Connection {self.connection_id}] Unrecognised message type: "
                                    f"{message_type!r} — {message}"
                                )

                        except (json.JSONDecodeError, UnicodeDecodeError):
                            # Not JSON - treat as binary audio data
                            is_json_message = False

                        # If not a JSON message, treat as audio data
                        if not is_json_message:
                            self.bytes_received += len(data)
                            remaining_count += 1

                            if self.should_playback:
                                self.audio_buffer.put(data)
                                logger.debug(f"[Connection {self.connection_id}] Received {len(data)} bytes of audio (total: {self.bytes_received})")
                            else:
                                logger.debug(f"[Connection {self.connection_id}] Received {len(data)} bytes of audio (discarded)")

                except asyncio.TimeoutError:
                    break

            if remaining_count > 0:
                logger.debug(f"[Connection {self.connection_id}] Processed {remaining_count} additional audio responses after stream close")

        except Exception as e:
            self._log_exception_details(e)

    async def end_session(self, force=False):
        """Close the streaming session

        Args:
            force: If True, cancel response task immediately without waiting for remaining audio
        """
        already_inactive = not self.is_active
        self.is_active = False
        self._flushed_event.set()  # Unblock any send loop waiting on wait_for_flushed()

        logger.debug(f"[Connection {self.connection_id}] Ending session (force={force})")

        # Close the input stream - this signals to Deepgram that no more text is coming
        try:
            if self.stream and self.stream.input_stream:
                await self.stream.input_stream.close()  # type: ignore
                logger.debug(f"[Connection {self.connection_id}] Input stream closed")
        except Exception as e:
            logger.debug(f"[Connection {self.connection_id}] Input stream close (may already be closed): {e}")

        # Wait for the response processing task to complete
        if self.response_task and not self.response_task.done():
            if force or already_inactive:
                # Shutdown requested — cancel immediately rather than waiting
                self.response_task.cancel()
                try:
                    await asyncio.wait_for(self.response_task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            else:
                try:
                    # Give it up to 30 seconds to finish processing remaining audio
                    # TTS can take longer than STT as it needs to synthesize all the text
                    await asyncio.wait_for(self.response_task, timeout=30.0)
                    logger.debug(f"[Connection {self.connection_id}] All audio received")
                except asyncio.TimeoutError:
                    logger.warning(f"[Connection {self.connection_id}] Timeout waiting for final audio (30s elapsed)")
                    self.response_task.cancel()
                except asyncio.CancelledError:
                    pass

        logger.info(f"[Connection {self.connection_id}] Session ended ({self.phrase_count} phrases, {self.bytes_received} bytes)")


class MultiConnectionTTSClient:
    """
    Client for streaming text to multiple simultaneous Deepgram TTS connections on AWS SageMaker.

    This client generates text and broadcasts it to multiple bidirectional streaming
    connections for load testing. Audio from a selected connection is played back
    to system speakers.
    """

    def __init__(self, endpoint_name, region=DEFAULT_REGION, num_connections=1, playback_connection_id=1):
        self.endpoint_name = endpoint_name
        self.region = region
        self.num_connections = num_connections
        self.playback_connection_id = playback_connection_id
        self.bidi_endpoint = f"https://runtime.sagemaker.{region}.amazonaws.com:8443"
        self.client = None
        self._boto_session = None
        self.connections = []
        self.is_active = False
        self.pyaudio_instance = None
        self.audio_stream = None
        self._ended = False

    def _initialize_client(self):
        """Initialize the SageMaker Runtime HTTP2 client with AWS credentials"""
        logger.debug("Initializing SageMaker client")

        # Use boto3 to resolve credentials via standard AWS credential chain
        try:
            session = boto3.Session(region_name=self.region)
            credentials = session.get_credentials()

            if credentials is None:
                raise NoCredentialsError()

            # Ensure credentials are available in environment for smithy client
            frozen_creds = credentials.get_frozen_credentials()
            os.environ['AWS_ACCESS_KEY_ID'] = frozen_creds.access_key
            os.environ['AWS_SECRET_ACCESS_KEY'] = frozen_creds.secret_key
            if frozen_creds.token:
                os.environ['AWS_SESSION_TOKEN'] = frozen_creds.token

            logger.debug("AWS credentials successfully loaded")

            # Optionally log the credential source for debugging
            caller_identity = session.client('sts').get_caller_identity()
            logger.debug(f"Authenticated as: {caller_identity.get('Arn', 'Unknown')}")

        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error("AWS credentials not found")
            logger.error("Please configure AWS credentials using one of these methods:")
            logger.error("  1. AWS CLI: aws configure")
            logger.error("  2. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
            logger.error("  3. AWS credentials file: ~/.aws/credentials")
            logger.error("  4. IAM role (when running on AWS infrastructure)")
            raise RuntimeError("AWS credentials not available") from e
        except Exception as e:
            logger.error(f"Error initializing AWS credentials: {e}")
            raise

        logger.debug(f"Using SageMaker endpoint: {self.bidi_endpoint}")
        logger.debug(f"Region: {self.region}")

        config = Config(
            endpoint_uri=self.bidi_endpoint,
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            auth_scheme_resolver=HTTPAuthSchemeResolver(),
            auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="sagemaker")}
        )
        self.client = SageMakerRuntimeHTTP2Client(config=config)
        self._boto_session = session
        logger.info("SageMaker client initialized successfully")

    def verify_endpoint(self):
        """
        Verify the SageMaker endpoint exists and is InService.

        Raises:
            RuntimeError: If the endpoint does not exist or is not InService.
        """
        if not self._boto_session:
            self._initialize_client()

        assert self._boto_session is not None
        sm_client = self._boto_session.client("sagemaker", region_name=self.region)
        try:
            response = sm_client.describe_endpoint(EndpointName=self.endpoint_name)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ValidationException":
                raise RuntimeError(
                    f"Endpoint '{self.endpoint_name}' does not exist in region '{self.region}'"
                ) from e
            raise

        status = response.get("EndpointStatus")
        if status != "InService":
            raise RuntimeError(
                f"Endpoint '{self.endpoint_name}' is not ready (status: {status})"
            )

        logger.info(f"Endpoint '{self.endpoint_name}' is InService")

    async def initialize_connections(self, voice="aura-2-thalia-en", **kwargs):
        """
        Initialize all streaming connections to SageMaker

        Args:
            voice: Deepgram TTS voice to use (default: aura-2-thalia-en)
            **kwargs: Additional Deepgram query parameters
        """
        if not self.client:
            self._initialize_client()

        logger.info(f"Initializing {self.num_connections} connection(s), playback on connection {self.playback_connection_id}...")

        # Create all connections
        for i in range(1, self.num_connections + 1):
            should_playback = (i == self.playback_connection_id)
            conn = DeepgramSageMakerTTSConnection(i, self.client, self.endpoint_name, should_playback=should_playback, boto_session=self._boto_session)
            self.connections.append(conn)

        # Start all sessions in parallel
        tasks = [
            conn.start_session(voice=voice, **kwargs)
            for conn in self.connections
        ]
        await asyncio.gather(*tasks)

        self.is_active = True
        logger.info(f"All {self.num_connections} connection(s) started successfully")

    def _initialize_audio_playback(self):
        """Initialize PyAudio for audio playback"""
        logger.info("Initializing audio playback device")

        self.pyaudio_instance = pyaudio.PyAudio()

        # List available audio output devices for debugging
        logger.debug("Available audio output devices:")
        for i in range(self.pyaudio_instance.get_device_count()):
            device_info = self.pyaudio_instance.get_device_info_by_index(i)
            if device_info['maxOutputChannels'] > 0:
                logger.debug(f"  [{i}] {device_info['name']}")

        try:
            self.audio_stream = self.pyaudio_instance.open(
                format=AUDIO_FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                output=True,
                frames_per_buffer=1024
            )
            logger.info("🔊 Audio playback device opened")

        except Exception as e:
            logger.error(f"Failed to open audio output: {e}")
            raise

    def _close_audio_playback(self):
        """Close audio playback"""
        if self.audio_stream:
            logger.debug("Stopping audio stream")
            self.audio_stream.stop_stream()
            self.audio_stream.close()

        if self.pyaudio_instance:
            logger.debug("Terminating PyAudio")
            self.pyaudio_instance.terminate()

        logger.info("Audio playback closed")

    async def stream_text_and_playback_audio(self, duration_seconds, phrases):
        """
        Stream text to all connections and playback audio from the selected connection

        Args:
            duration_seconds: How long to run the test for in seconds
            phrases: List of text phrases to cycle through and send
        """
        if self.playback_connection_id <= self.num_connections:
            self._initialize_audio_playback()

        playback_conn = self.connections[self.playback_connection_id - 1] if self.playback_connection_id <= self.num_connections else None

        start_time = None  # Set when first audio chunk is played
        phrase_index = 0
        send_loop_done = False

        async def send_text_loop():
            """Continuously send text to all connections for the specified duration"""
            nonlocal phrase_index, send_loop_done
            try:
                while self.is_active:
                    # Check if duration has elapsed since first audio playback
                    if start_time is not None:
                        elapsed = time.time() - start_time
                        if elapsed >= duration_seconds:
                            logger.info(f"Duration elapsed ({duration_seconds}s), stopping text generation")
                            break

                    active_connections = [conn for conn in self.connections if conn.is_active]
                    if not active_connections:
                        logger.error("All connections have failed, stopping text generation")
                        break

                    # Wait for every active connection to acknowledge the previous Flush
                    # before sending the next phrase. This keeps the pipeline in sync and
                    # prevents exceeding the server-side flush rate limit.
                    await asyncio.gather(*[conn.wait_for_flushed() for conn in active_connections])

                    # Re-check after waiting — duration may have elapsed or connections failed
                    if not self.is_active:
                        break
                    if start_time is not None and time.time() - start_time >= duration_seconds:
                        logger.info(f"Duration elapsed ({duration_seconds}s), stopping text generation")
                        break

                    active_connections = [conn for conn in self.connections if conn.is_active]
                    if not active_connections:
                        logger.error("All connections have failed, stopping text generation")
                        break

                    phrase = phrases[phrase_index % len(phrases)]
                    phrase_index += 1

                    # Send text then Flush to trigger audio synthesis.
                    # Per Deepgram docs, audio is only generated after a Flush.
                    await asyncio.gather(*[conn.send_text(phrase) for conn in active_connections])
                    await asyncio.gather(*[
                        conn.send_flush()
                        for conn in active_connections
                        if conn.is_active and not conn.close_sent
                    ])

            except Exception as e:
                logger.error(f"Error in text sending loop: {e}")
                raise
            finally:
                send_loop_done = True
                logger.debug("Text sending loop completed")
                # Forcibly stop the audio stream so any in-progress write()
                # call on the playback thread is interrupted immediately
                if self.audio_stream:
                    try:
                        self.audio_stream.stop_stream()
                    except Exception:
                        pass

        async def playback_audio_loop():
            """Process and playback audio from the selected connection"""
            nonlocal start_time
            loop = asyncio.get_running_loop()
            try:
                while self.is_active and playback_conn:
                    if not playback_conn.audio_buffer.empty():
                        audio_data = playback_conn.audio_buffer.get()
                        if start_time is None:
                            start_time = time.time()
                            logger.info("First audio received, duration timer started")
                        try:
                            # Run blocking PyAudio write on a thread so the event loop
                            # (and the duration timer) remain responsive during playback
                            await loop.run_in_executor(
                                None, self.audio_stream.write, audio_data
                            )
                            logger.debug(f"Played {len(audio_data)} bytes of audio")
                        except Exception as e:
                            if send_loop_done:
                                # Stream was stopped by the duration expiry — exit cleanly
                                logger.debug("Audio write interrupted by stream stop")
                                return
                            logger.error(f"Error writing audio: {e}")
                    elif send_loop_done:
                        # Duration elapsed — stop playback immediately
                        logger.debug("Send loop done, stopping audio playback")
                        break
                    else:
                        # Small delay to prevent busy waiting
                        await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in playback loop: {e}")
                raise

        # Run both loops concurrently
        if playback_conn:
            await asyncio.gather(
                send_text_loop(),
                playback_audio_loop()
            )
        else:
            # No playback connection — start timer immediately when sending begins
            start_time = time.time()
            await send_text_loop()

    async def end_all_sessions(self, force=False):
        """Close all streaming sessions

        Args:
            force: If True, cancel response tasks immediately (used on Ctrl+C)
        """
        if self._ended:
            return
        self._ended = True

        logger.debug(f"Ending all sessions (force={force})")
        self.is_active = False

        # Close audio playback first
        self._close_audio_playback()

        # End all connection sessions in parallel
        logger.info(f"Closing {len(self.connections)} connection(s)...")
        tasks = [conn.end_session(force=force) for conn in self.connections]
        await asyncio.gather(*tasks)

        logger.info("All sessions ended successfully")


async def main():
    """Main function to run the multi-connection Deepgram TTS streaming client"""
    parser = argparse.ArgumentParser(
        description="Stream text to multiple simultaneous Deepgram TTS connections on SageMaker"
    )
    parser.add_argument(
        "endpoint_name",
        help="SageMaker endpoint name"
    )
    parser.add_argument(
        "--connections",
        type=int,
        default=1,
        help="Number of simultaneous streaming connections (default: 1)"
    )
    parser.add_argument(
        "--playback",
        type=int,
        default=1,
        help="Connection ID to playback audio to speakers (default: 1)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="How long to run the test in seconds (default: 30)"
    )
    parser.add_argument(
        "--voice",
        default="aura-2-thalia-en",
        help="Deepgram TTS voice to use (default: aura-2-thalia-en)"
    )
    parser.add_argument(
        "--text-file",
        default="tts-input.txt",
        help="Path to file containing text phrases to synthesize, one per line (default: tts-input.txt)"
    )
    parser.add_argument(
        "--region",
        default="us-east-2",
        help="AWS region (default: us-east-2)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Validate parameters
    if args.connections < 1:
        print("ERROR: --connections must be a positive integer (minimum 1)")
        sys.exit(1)

    if args.playback < 1 or args.playback > args.connections:
        print(f"ERROR: --playback must be between 1 and {args.connections}")
        sys.exit(1)

    if args.duration < 1:
        print("ERROR: --duration must be a positive integer (minimum 1 second)")
        sys.exit(1)

    # Load text phrases from file
    try:
        with open(args.text_file, "r", encoding="utf-8") as f:
            phrases = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"ERROR: Text file not found: {args.text_file}")
        sys.exit(1)
    except OSError as e:
        print(f"ERROR: Could not read text file '{args.text_file}': {e}")
        sys.exit(1)

    if not phrases:
        print(f"ERROR: Text file '{args.text_file}' contains no non-empty lines")
        sys.exit(1)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

    print("=" * 60)
    print("Deepgram SageMaker Multi-Connection TTS Streaming Client")
    print("=" * 60)
    print(f"Endpoint: {args.endpoint_name}")
    print(f"Connections: {args.connections}")
    print(f"Playback Connection: {args.playback}")
    print(f"Duration: {args.duration} seconds")
    print(f"Voice: {args.voice}")
    print(f"Region: {args.region}")
    print(f"Text file: {args.text_file} ({len(phrases)} phrase(s))")
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Channels: {CHANNELS} (Mono)")
    print("=" * 60)

    # Create client
    client = MultiConnectionTTSClient(
        endpoint_name=args.endpoint_name,
        region=args.region,
        num_connections=args.connections,
        playback_connection_id=args.playback
    )

    loop = asyncio.get_running_loop()
    streaming_task = None
    shutdown_event = asyncio.Event()

    def handle_sigint():
        if not shutdown_event.is_set():
            print("\n\nReceived interrupt signal, stopping...")
            shutdown_event.set()
            # Mark all connections inactive so their loops exit
            client.is_active = False
            for conn in client.connections:
                conn.is_active = False
                if conn.response_task and not conn.response_task.done():
                    conn.response_task.cancel()
            # Cancel the streaming task so stream_text_and_playback_audio exits
            if streaming_task and not streaming_task.done():
                streaming_task.cancel()

    loop.add_signal_handler(signal.SIGINT, handle_sigint)

    exit_code = 0
    try:
        # Verify the endpoint exists and is ready before opening connections
        client._initialize_client()
        client.verify_endpoint()

        # Initialize all connections with Deepgram parameters
        await client.initialize_connections(voice=args.voice)

        print("\n" + "="*60)
        print(f"🎧 TTS STREAMING - {args.connections} Connection(s)")
        print(f"   Audio playback: Connection {args.playback}")
        print(f"   Duration: {args.duration} seconds")
        print("   (Press Ctrl+C to stop)")
        print("="*60 + "\n")

        # Stream text and playback audio
        streaming_task = asyncio.create_task(
            client.stream_text_and_playback_audio(args.duration, phrases)
        )
        await streaming_task

    except asyncio.CancelledError:
        logger.info("Streaming cancelled by user")
    except Exception as e:
        logger.error(f"Error during streaming: {e}", exc_info=True)
        exit_code = 1
    finally:
        loop.remove_signal_handler(signal.SIGINT)
        print("\n" + "="*60)
        if not shutdown_event.is_set():
            await client.end_all_sessions(force=True)

    if not shutdown_event.is_set() and exit_code == 0:
        logger.info("✅ TTS streaming complete!")
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
