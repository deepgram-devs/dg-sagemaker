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
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
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

# Test corpus - a few words to stream repeatedly
TEST_PHRASES = [
    "Hello world",
    "Testing text to speech",
    "Streaming audio data",
    "Multiple connections",
    "SageMaker Deepgram integration",
]

logger = logging.getLogger(__name__)


class DeepgramSageMakerTTSConnection:
    """
    Represents a single bidirectional streaming connection to SageMaker for TTS.

    Each connection handles its own stream and audio response processing.
    """

    def __init__(self, connection_id, client, endpoint_name, should_playback=False):
        self.connection_id = connection_id
        self.client = client
        self.endpoint_name = endpoint_name
        self.should_playback = should_playback
        self.stream = None
        self.output_stream = None
        self.is_active = False
        self.response_task = None
        self.audio_buffer = Queue()
        self.bytes_received = 0
        self.phrase_count = 0

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
        if not self.is_active:
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
            logger.debug(f"[Connection {self.connection_id}] Sent phrase {self.phrase_count}: '{text}'")
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
            logger.info(f"[Connection {self.connection_id}] Sent Close message")
            return True
        except Exception as e:
            logger.error(f"[Connection {self.connection_id}] Error sending Close message: {e}")
            return False

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
                            logger.debug(f"[Connection {self.connection_id}] Received metadata: "
                                       f"model={message.get('model_name')}, "
                                       f"request_id={message.get('request_id')}")

                        elif message_type == 'Warning':
                            logger.warning(f"[Connection {self.connection_id}] API Warning: {message.get('description')} (code: {message.get('code')})")

                        elif message_type in ('Flushed', 'Cleared'):
                            logger.debug(f"[Connection {self.connection_id}] Received {message_type} message (seq: {message.get('sequence_id')})")

                        else:
                            logger.debug(f"[Connection {self.connection_id}] Received unknown message type: {message_type}")

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
                                logger.debug(f"[Connection {self.connection_id}] Received metadata: "
                                           f"model={message.get('model_name')}, "
                                           f"request_id={message.get('request_id')}")
                            elif message_type == 'Warning':
                                logger.warning(f"[Connection {self.connection_id}] API Warning: {message.get('description')}")

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
            logger.error(f"[Connection {self.connection_id}] Error processing audio responses: {e}", exc_info=True)

    async def end_session(self):
        """Close the streaming session"""
        if not self.is_active:
            return

        logger.debug(f"[Connection {self.connection_id}] Ending session")
        self.is_active = False

        # Close the input stream - this signals to Deepgram that no more text is coming
        try:
            if self.stream and self.stream.input_stream:
                await self.stream.input_stream.close()  # type: ignore
                logger.debug(f"[Connection {self.connection_id}] Input stream closed, waiting for final audio")
        except Exception as e:
            logger.debug(f"[Connection {self.connection_id}] Input stream close (may already be closed): {e}")

        # Wait for the response processing task to complete naturally
        if self.response_task and not self.response_task.done():
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

        logger.info(f"[Connection {self.connection_id}] Session ended successfully ({self.phrase_count} phrases, {self.bytes_received} bytes)")


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
        self.connections = []
        self.is_active = False
        self.pyaudio_instance = None
        self.audio_stream = None

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
        logger.info("SageMaker client initialized successfully")

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
            conn = DeepgramSageMakerTTSConnection(i, self.client, self.endpoint_name, should_playback=should_playback)
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

    async def stream_text_and_playback_audio(self, duration_seconds):
        """
        Stream text to all connections and playback audio from the selected connection

        Args:
            duration_seconds: How long to run the test for in seconds
        """
        if self.playback_connection_id <= self.num_connections:
            self._initialize_audio_playback()

        playback_conn = self.connections[self.playback_connection_id - 1] if self.playback_connection_id <= self.num_connections else None

        start_time = time.time()
        phrase_index = 0
        send_loop_done = False

        async def send_text_loop():
            """Continuously send text to all connections for the specified duration"""
            nonlocal phrase_index, send_loop_done
            try:
                while self.is_active:
                    # Check if duration has elapsed
                    elapsed = time.time() - start_time
                    if elapsed >= duration_seconds:
                        logger.info(f"Duration elapsed ({duration_seconds}s), stopping text generation")
                        break

                    # Check if all connections have errors
                    active_connections = [conn for conn in self.connections if conn.is_active]
                    if not active_connections:
                        logger.error("All connections have failed, stopping text generation")
                        break

                    phrase = TEST_PHRASES[phrase_index % len(TEST_PHRASES)]
                    phrase_index += 1

                    # Send to all active connections in parallel
                    tasks = [
                        conn.send_text(phrase)
                        for conn in active_connections
                    ]
                    if tasks:
                        await asyncio.gather(*tasks)

                    # Small delay between phrases
                    await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error in text sending loop: {e}")
                raise
            finally:
                send_loop_done = True
                logger.debug("Text sending loop completed")

        async def playback_audio_loop():
            """Process and playback audio from the selected connection"""
            try:
                no_data_timeout = 2.0  # seconds
                last_data_time = time.time()

                while self.is_active and playback_conn:
                    if not playback_conn.audio_buffer.empty():
                        audio_data = playback_conn.audio_buffer.get()
                        try:
                            self.audio_stream.write(audio_data)
                            logger.debug(f"Played {len(audio_data)} bytes of audio")
                        except Exception as e:
                            logger.error(f"Error writing audio: {e}")
                        last_data_time = time.time()
                    elif send_loop_done and playback_conn.response_task and playback_conn.response_task.done():
                        # Send loop is done, response processor is done, and buffer is empty - no more audio coming
                        logger.debug(f"All audio received and played from connection {playback_conn.connection_id}")
                        break
                    elif send_loop_done and (time.time() - last_data_time > no_data_timeout):
                        # Send loop is done and no data received for timeout period, assume stream is done
                        logger.debug(f"No audio data received for {no_data_timeout}s after send complete, exiting playback loop")
                        break
                    else:
                        # Small delay to prevent busy waiting
                        await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in playback loop: {e}")
                raise

        # Run both loops concurrently
        try:
            if playback_conn:
                await asyncio.gather(
                    send_text_loop(),
                    playback_audio_loop()
                )
            else:
                await send_text_loop()
        finally:
            # After duration expires, send Close message to all connections
            # This signals to Deepgram to finalize synthesis according to API spec
            logger.info("Sending Close message to all connections")
            close_tasks = [
                conn.send_close_message()
                for conn in self.connections
                if conn.is_active
            ]
            if close_tasks:
                await asyncio.gather(*close_tasks)

            # Small delay to allow Close message to be processed
            await asyncio.sleep(0.1)

            # Close input streams to signal end of transmission
            logger.info("Closing all input streams")
            for conn in self.connections:
                if conn.is_active:
                    try:
                        await conn.stream.input_stream.close()  # type: ignore
                        logger.debug(f"[Connection {conn.connection_id}] Input stream closed")
                    except Exception as e:
                        logger.debug(f"[Connection {conn.connection_id}] Error closing input stream: {e}")

            # Close output streams
            logger.info("Closing all output streams")
            for conn in self.connections:
                if conn.output_stream:
                    try:
                        await conn.output_stream.aclose()
                        logger.debug(f"[Connection {conn.connection_id}] Output stream closed")
                    except Exception as e:
                        logger.debug(f"[Connection {conn.connection_id}] Error closing output stream: {e}")

    async def end_all_sessions(self):
        """Close all streaming sessions"""
        if not self.is_active:
            return

        logger.debug("Ending all sessions")
        self.is_active = False

        # Close audio playback first
        self._close_audio_playback()

        # End all connection sessions in parallel
        logger.info(f"Closing {len(self.connections)} connection(s)...")
        tasks = [conn.end_session() for conn in self.connections]
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

    # Handle Ctrl+C gracefully
    def signal_handler(_sig, _frame):
        print("\n\nReceived interrupt signal, stopping...")
        client.is_active = False

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Initialize all connections with Deepgram parameters
        await client.initialize_connections(voice=args.voice)

        print("\n" + "="*60)
        print(f"🎧 TTS STREAMING - {args.connections} Connection(s)")
        print(f"   Audio playback: Connection {args.playback}")
        print(f"   Duration: {args.duration} seconds")
        print("   (Press Ctrl+C to stop)")
        print("="*60 + "\n")

        # Stream text and playback audio
        await client.stream_text_and_playback_audio(args.duration)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during streaming: {e}", exc_info=True)
        return 1
    finally:
        # Clean up
        await client.end_all_sessions()
        print("\n" + "="*60)

    logger.info("✅ TTS streaming complete!")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
