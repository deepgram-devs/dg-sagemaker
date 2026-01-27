#!/usr/bin/env python3
"""
Deepgram SageMaker Microphone Streaming Client - Multiple Connections

This client captures live audio from your microphone and streams it to multiple
simultaneous bidirectional connections to a Deepgram model deployed on SageMaker
for real-time transcription. Useful for load testing and stress testing endpoints.
"""

import asyncio
import argparse
import json
import logging
import os
import signal
import sys
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
    print("ERROR: pyaudio is required for microphone input")
    print("Install it with: pip install pyaudio")
    print("On macOS, you may need: brew install portaudio && pip install pyaudio")
    sys.exit(1)

# Configuration constants
DEFAULT_REGION = "us-east-2"
AUDIO_FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1  # Mono audio
SAMPLE_RATE = 16000  # 16kHz sample rate (good for speech recognition)
CHUNK_SIZE = 8192  # Bytes per audio chunk

logger = logging.getLogger(__name__)


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

    async def start_session(self, model="nova-3", language="en", **kwargs):
        """
        Start a bidirectional streaming session with Deepgram on SageMaker

        Args:
            model: Deepgram model to use (default: nova-3)
            language: Language code (default: en)
            **kwargs: Additional Deepgram query parameters (diarize, punctuate, etc.)
        """
        # Build query string for Deepgram parameters
        query_params = {
            "model": model,
            "language": language,
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
            model_invocation_path="v1/listen",
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
        self.response_task = asyncio.create_task(self._process_responses())

        # Give the response processor a moment to start
        await asyncio.sleep(0.1)

        logger.info(f"[Connection {self.connection_id}] Session started successfully")

    async def send_audio_chunk(self, audio_bytes):
        """Send a chunk of audio data to the stream"""
        if not self.is_active:
            return

        try:
            payload = RequestPayloadPart(bytes_=audio_bytes)
            event = RequestStreamEventPayloadPart(value=payload)
            await self.stream.input_stream.send(event)
            self.chunk_count += 1
            logger.debug(f"[Connection {self.connection_id}] Sent chunk {self.chunk_count} ({len(audio_bytes)} bytes)")
        except Exception as e:
            logger.error(f"[Connection {self.connection_id}] Error sending audio chunk: {e}")
            self.is_active = False

    async def _process_responses(self):
        """Process streaming responses from Deepgram"""
        try:
            logger.debug(f"[Connection {self.connection_id}] Response processor started")

            while self.is_active:
                result = await self.output_stream.receive()

                if result is None:
                    logger.debug(f"[Connection {self.connection_id}] No more responses from server")
                    break

                if result.value and result.value.bytes_:
                    response_data = result.value.bytes_.decode('utf-8')

                    try:
                        # Parse JSON response from Deepgram
                        parsed = json.loads(response_data)

                        # Extract and print transcript if available
                        if 'channel' in parsed:
                            alternatives = parsed.get('channel', {}).get('alternatives', [])
                            if alternatives and alternatives[0].get('transcript'):
                                transcript = alternatives[0]['transcript']
                                if transcript.strip():  # Only print non-empty transcripts
                                    confidence = alternatives[0].get('confidence', 0)
                                    is_final = parsed.get('is_final', False)
                                    speech_final = parsed.get('speech_final', False)

                                    # Show final vs interim results differently
                                    if is_final and speech_final:
                                        print(f"[Conn {self.connection_id}] ✓ {transcript} ({confidence:.1%})")
                                    else:
                                        print(f"[Conn {self.connection_id}]   {transcript} [interim]")

                    except json.JSONDecodeError:
                        logger.warning(f"[Connection {self.connection_id}] Non-JSON response: {response_data}")

            # Continue processing any remaining buffered responses
            logger.debug(f"[Connection {self.connection_id}] Processing any remaining buffered responses...")
            remaining_count = 0
            while remaining_count < 10:
                try:
                    result = await asyncio.wait_for(self.output_stream.receive(), timeout=0.5)
                    if result is None:
                        break

                    if result.value and result.value.bytes_:
                        remaining_count += 1
                        response_data = result.value.bytes_.decode('utf-8')
                        try:
                            parsed = json.loads(response_data)
                            if 'channel' in parsed:
                                alternatives = parsed.get('channel', {}).get('alternatives', [])
                                if alternatives and alternatives[0].get('transcript'):
                                    transcript = alternatives[0]['transcript']
                                    if transcript.strip():
                                        confidence = alternatives[0].get('confidence', 0)
                                        is_final = parsed.get('is_final', False)
                                        speech_final = parsed.get('speech_final', False)
                                        if is_final and speech_final:
                                            print(f"[Conn {self.connection_id}] ✓ {transcript} ({confidence:.1%})")
                                        else:
                                            print(f"[Conn {self.connection_id}]   {transcript} [interim]")
                        except json.JSONDecodeError:
                            pass
                except asyncio.TimeoutError:
                    break

            if remaining_count > 0:
                logger.debug(f"[Connection {self.connection_id}] Processed {remaining_count} additional responses after stream close")

        except Exception as e:
            logger.error(f"[Connection {self.connection_id}] Error processing responses: {e}", exc_info=True)

    async def end_session(self):
        """Close the streaming session"""
        if not self.is_active:
            return

        logger.debug(f"[Connection {self.connection_id}] Ending session")
        self.is_active = False

        # Close the input stream - this signals to Deepgram that no more audio is coming
        try:
            await self.stream.input_stream.close()
            logger.debug(f"[Connection {self.connection_id}] Input stream closed, waiting for final responses")
        except Exception as e:
            logger.error(f"[Connection {self.connection_id}] Error closing input stream: {e}")

        # Wait for the response processing task to complete naturally
        if self.response_task and not self.response_task.done():
            try:
                # Give it up to 15 seconds to finish processing remaining responses
                await asyncio.wait_for(self.response_task, timeout=15.0)
                logger.debug(f"[Connection {self.connection_id}] All responses received")
            except asyncio.TimeoutError:
                logger.warning(f"[Connection {self.connection_id}] Timeout waiting for final responses (15s elapsed)")
                self.response_task.cancel()
            except asyncio.CancelledError:
                pass

        logger.info(f"[Connection {self.connection_id}] Session ended successfully (sent {self.chunk_count} chunks)")


class MultiConnectionMicrophoneClient:
    """
    Client for streaming microphone audio to multiple simultaneous Deepgram connections on AWS SageMaker.

    This client uses PyAudio to capture live microphone input and broadcasts it
    to multiple bidirectional streaming connections for load testing.
    """

    def __init__(self, endpoint_name, region=DEFAULT_REGION, num_connections=1):
        self.endpoint_name = endpoint_name
        self.region = region
        self.num_connections = num_connections
        self.bidi_endpoint = f"https://runtime.sagemaker.{region}.amazonaws.com:8443"
        self.client = None
        self.connections = []
        self.is_active = False
        self.audio_queue = Queue()
        self.pyaudio_instance = None
        self.audio_stream = None
        self.stream_tasks = []

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

    async def initialize_connections(self, model="nova-3", language="en", **kwargs):
        """
        Initialize all streaming connections to SageMaker

        Args:
            model: Deepgram model to use (default: nova-3)
            language: Language code (default: en)
            **kwargs: Additional Deepgram query parameters (diarize, punctuate, etc.)
        """
        if not self.client:
            self._initialize_client()

        logger.info(f"Initializing {self.num_connections} connection(s)...")

        # Create all connections
        for i in range(self.num_connections):
            conn = DeepgramSageMakerConnection(i + 1, self.client, self.endpoint_name)
            self.connections.append(conn)

        # Start all sessions in parallel
        tasks = [
            conn.start_session(model=model, language=language, **kwargs)
            for conn in self.connections
        ]
        await asyncio.gather(*tasks)

        self.is_active = True
        logger.info(f"All {self.num_connections} connection(s) started successfully")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for PyAudio to handle incoming audio data"""
        if status:
            logger.warning(f"PyAudio status: {status}")

        # Put audio data in queue for async processing
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    async def start_microphone(self):
        """Initialize and start capturing audio from the microphone"""
        logger.info("Initializing microphone")

        self.pyaudio_instance = pyaudio.PyAudio()

        # List available audio devices for debugging
        logger.debug("Available audio input devices:")
        for i in range(self.pyaudio_instance.get_device_count()):
            device_info = self.pyaudio_instance.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                logger.debug(f"  [{i}] {device_info['name']}")

        try:
            self.audio_stream = self.pyaudio_instance.open(
                format=AUDIO_FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=self._audio_callback
            )

            logger.info("🎤 Microphone started - speak now!")
            self.audio_stream.start_stream()

        except Exception as e:
            logger.error(f"Failed to open microphone: {e}")
            raise

    async def stream_microphone_audio(self):
        """
        Stream microphone audio to all connections in real-time

        Creates a separate task for each connection to send audio chunks.
        """
        async def send_to_connection(conn):
            """Task to send audio chunks to a specific connection"""
            chunks_sent = 0
            while self.is_active and conn.is_active:
                # Check if there's audio data in the queue
                if not self.audio_queue.empty():
                    # Peek at the audio chunk (we need to send to all connections)
                    # Since Queue doesn't have peek, we'll handle this differently
                    await asyncio.sleep(0.001)
                else:
                    # Small delay to prevent busy waiting
                    await asyncio.sleep(0.01)

            logger.debug(f"[Connection {conn.connection_id}] Streaming task ended")

        # Create a task that broadcasts audio to all connections
        async def broadcast_audio():
            """Broadcast audio chunks from queue to all active connections"""
            chunk_count = 0
            try:
                while self.is_active:
                    if not self.audio_queue.empty():
                        audio_chunk = self.audio_queue.get()
                        chunk_count += 1

                        # Send this chunk to all active connections in parallel
                        tasks = [
                            conn.send_audio_chunk(audio_chunk)
                            for conn in self.connections
                            if conn.is_active
                        ]
                        if tasks:
                            await asyncio.gather(*tasks)

                        logger.debug(f"Broadcast chunk {chunk_count} to {len(tasks)} connection(s)")
                    else:
                        # Small delay to prevent busy waiting
                        await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error broadcasting audio: {e}")
                raise
            finally:
                logger.info(f"Broadcasted {chunk_count} audio chunks total")

        # Run the broadcast task
        await broadcast_audio()

    def stop_microphone(self):
        """Stop capturing audio from the microphone"""
        if self.audio_stream:
            logger.debug("Stopping microphone stream")
            self.audio_stream.stop_stream()
            self.audio_stream.close()

        if self.pyaudio_instance:
            logger.debug("Terminating PyAudio")
            self.pyaudio_instance.terminate()

        logger.info("Microphone stopped")

    async def end_all_sessions(self):
        """Close all streaming sessions"""
        if not self.is_active:
            return

        logger.debug("Ending all sessions")
        self.is_active = False

        # Stop the microphone first
        self.stop_microphone()

        # End all connection sessions in parallel
        logger.info(f"Closing {len(self.connections)} connection(s)...")
        tasks = [conn.end_session() for conn in self.connections]
        await asyncio.gather(*tasks)

        logger.info("All sessions ended successfully")


async def main():
    """Main function to run the multi-connection Deepgram microphone streaming client"""
    parser = argparse.ArgumentParser(
        description="Stream microphone audio to multiple simultaneous Deepgram connections on SageMaker"
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
        "--model",
        default="nova-3",
        help="Deepgram model to use (default: nova-3)"
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code (default: en)"
    )
    parser.add_argument(
        "--diarize",
        default="false",
        help="Enable speaker diarization (default: false)"
    )
    parser.add_argument(
        "--punctuate",
        default="true",
        help="Enable punctuation (default: true)"
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (default: us-east-1)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Validate connections parameter
    if args.connections < 1:
        print("ERROR: --connections must be a positive integer (minimum 1)")
        sys.exit(1)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

    print("=" * 60)
    print("Deepgram SageMaker Multi-Connection Streaming Client")
    print("=" * 60)
    print(f"Endpoint: {args.endpoint_name}")
    print(f"Connections: {args.connections}")
    print(f"Model: {args.model}")
    print(f"Language: {args.language}")
    print(f"Region: {args.region}")
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Channels: {CHANNELS} (Mono)")
    print("=" * 60)

    # Create client
    client = MultiConnectionMicrophoneClient(
        endpoint_name=args.endpoint_name,
        region=args.region,
        num_connections=args.connections
    )

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\nReceived interrupt signal, stopping...")
        client.is_active = False

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Initialize all connections with Deepgram parameters
        await client.initialize_connections(
            model=args.model,
            language=args.language,
            diarize=args.diarize,
            punctuate=args.punctuate
        )

        print("\n" + "="*60)
        print(f"🎤 LIVE TRANSCRIPTION - {args.connections} Connection(s)")
        print("   (Press Ctrl+C to stop)")
        print("="*60 + "\n")

        # Start microphone capture
        await client.start_microphone()

        # Stream microphone audio until interrupted
        await client.stream_microphone_audio()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during streaming: {e}", exc_info=True)
        return 1
    finally:
        # Clean up
        await client.end_all_sessions()
        print("\n" + "="*60)

    logger.info("✅ Transcription complete!")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
