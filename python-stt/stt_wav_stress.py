#!/usr/bin/env python3
"""
Deepgram SageMaker WAV File Streaming Client - Multiple Connections

This client streams a WAV audio file in real-time to multiple simultaneous
bidirectional connections to a Deepgram model deployed on SageMaker for
real-time transcription. Audio is paced to match the WAV file's sample rate
so it simulates a live microphone source. Useful for repeatable load testing
and stress testing endpoints.
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
from urllib.parse import quote
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

# Configuration constants
DEFAULT_REGION = "us-east-2"
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

    async def start_session(self, sample_rate, model="nova-3", language="en", **kwargs):
        """
        Start a bidirectional streaming session with Deepgram on SageMaker

        Args:
            sample_rate: Sample rate of the audio being sent
            model: Deepgram model to use (default: nova-3)
            language: Language code (default: en)
            **kwargs: Additional Deepgram query parameters (diarize, punctuate, etc.)
        """
        keywords_list = kwargs.pop('keywords', [])

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

        logger.debug(f"[Connection {self.connection_id}] Starting session with endpoint: {self.endpoint_name}")
        logger.debug(f"[Connection {self.connection_id}] Deepgram parameters: {query_string}")

        stream_input = InvokeEndpointWithBidirectionalStreamInput(
            endpoint_name=self.endpoint_name,
            model_invocation_path="v1/listen",
            model_query_string=query_string
        )

        logger.debug(f"[Connection {self.connection_id}] Invoking endpoint with bidirectional stream")
        self.stream = await self.client.invoke_endpoint_with_bidirectional_stream(stream_input)
        self.is_active = True

        logger.debug(f"[Connection {self.connection_id}] Stream created, connecting to output stream")
        output = await self.stream.await_output()
        self.output_stream = output[1]
        logger.debug(f"[Connection {self.connection_id}] Connected to output stream")

        logger.debug(f"[Connection {self.connection_id}] Starting response processor")
        self.response_task = asyncio.create_task(self._process_responses())

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
                        logger.warning(f"[Connection {self.connection_id}] Non-JSON response: {response_data}")

            # Drain any remaining buffered responses
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

        try:
            await self.stream.input_stream.close()
            logger.debug(f"[Connection {self.connection_id}] Input stream closed, waiting for final responses")
        except Exception as e:
            logger.error(f"[Connection {self.connection_id}] Error closing input stream: {e}")

        if self.response_task and not self.response_task.done():
            try:
                await asyncio.wait_for(self.response_task, timeout=15.0)
                logger.debug(f"[Connection {self.connection_id}] All responses received")
            except asyncio.TimeoutError:
                logger.warning(f"[Connection {self.connection_id}] Timeout waiting for final responses (15s elapsed)")
                self.response_task.cancel()
            except asyncio.CancelledError:
                pass

        logger.info(f"[Connection {self.connection_id}] Session ended successfully (sent {self.chunk_count} chunks)")


class MultiConnectionWAVClient:
    """
    Client for streaming a WAV file in real-time to multiple simultaneous
    Deepgram connections on AWS SageMaker.

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

        # WAV metadata filled in by _open_wav
        self.sample_rate = None
        self.channels = None
        self.sample_width = None  # bytes per sample
        self.duration_seconds = None

    def _open_wav(self):
        """Read WAV header and validate the file, returning a wave.Wave_read object."""
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
        total_frames = wf.getnframes()
        self.duration_seconds = total_frames / self.sample_rate

        if self.sample_width != 2:
            wf.close()
            raise ValueError(
                f"WAV file must be 16-bit PCM (sample width 2 bytes). "
                f"Got {self.sample_width * 8}-bit audio."
            )

        logger.info(
            f"WAV file: {self.wav_path} | "
            f"{self.sample_rate} Hz | "
            f"{self.channels}ch | "
            f"{self.duration_seconds:.2f}s"
        )
        return wf

    def _initialize_client(self):
        """Initialize the SageMaker Runtime HTTP2 client with AWS credentials"""
        logger.debug("Initializing SageMaker client")

        try:
            session = boto3.Session(region_name=self.region)
            credentials = session.get_credentials()

            if credentials is None:
                raise NoCredentialsError()

            frozen_creds = credentials.get_frozen_credentials()
            os.environ['AWS_ACCESS_KEY_ID'] = frozen_creds.access_key
            os.environ['AWS_SECRET_ACCESS_KEY'] = frozen_creds.secret_key
            if frozen_creds.token:
                os.environ['AWS_SESSION_TOKEN'] = frozen_creds.token

            logger.debug("AWS credentials successfully loaded")

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

        for i in range(self.num_connections):
            conn = DeepgramSageMakerConnection(i + 1, self.client, self.endpoint_name)
            self.connections.append(conn)

        tasks = [
            conn.start_session(self.sample_rate, model=model, language=language, **kwargs)
            for conn in self.connections
        ]
        await asyncio.gather(*tasks)

        self.is_active = True
        logger.info(f"All {self.num_connections} connection(s) started successfully")

    async def stream_wav_audio(self, loop=False):
        """
        Read the WAV file and broadcast audio chunks to all connections at
        real-time speed, paced by the file's sample rate.

        Args:
            loop: If True, restart the file when it ends (until is_active=False).
        """
        bytes_per_frame = self.sample_width * self.channels
        # How many frames fit in one CHUNK_SIZE buffer
        frames_per_chunk = CHUNK_SIZE // bytes_per_frame
        # Real-time duration of one chunk in seconds
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
                    break  # End of file

                # Broadcast to all active connections
                tasks = [
                    conn.send_audio_chunk(raw)
                    for conn in self.connections
                    if conn.is_active
                ]
                if tasks:
                    await asyncio.gather(*tasks)
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

    async def end_all_sessions(self):
        """Close all streaming sessions"""
        if not self.is_active:
            return

        logger.debug("Ending all sessions")
        self.is_active = False

        logger.info(f"Closing {len(self.connections)} connection(s)...")
        tasks = [conn.end_session() for conn in self.connections]
        await asyncio.gather(*tasks)

        logger.info("All sessions ended successfully")


async def main():
    """Main function to run the multi-connection Deepgram WAV streaming client"""
    parser = argparse.ArgumentParser(
        description="Stream a WAV file in real-time to multiple simultaneous Deepgram connections on SageMaker"
    )
    parser.add_argument(
        "endpoint_name",
        help="SageMaker endpoint name"
    )
    parser.add_argument(
        "--file",
        required=True,
        metavar="WAV_FILE",
        help="Path to the WAV file to stream (must be 16-bit PCM)"
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
    parser.add_argument(
        "--keywords",
        default="",
        help="Comma-delimited keywords in format 'keyword:intensity' (e.g., 'hello:5,world:10')"
    )
    parser.add_argument(
        "--interim-results",
        default="true",
        choices=["true", "false"],
        help="Enable or disable interim results (default: true)"
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the WAV file continuously until Ctrl+C or --duration is reached"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration in seconds to run before stopping automatically (default: run until file ends or Ctrl+C)"
    )

    args = parser.parse_args()

    # Parse keywords parameter
    keywords_list = []
    if args.keywords:
        for keyword_item in args.keywords.split(','):
            keyword_item = keyword_item.strip()
            if keyword_item:
                keywords_list.append(keyword_item)

    if args.connections < 1:
        print("ERROR: --connections must be a positive integer (minimum 1)")
        sys.exit(1)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

    # Create client and open WAV to validate + read metadata before printing banner
    client = MultiConnectionWAVClient(
        endpoint_name=args.endpoint_name,
        wav_path=args.file,
        region=args.region,
        num_connections=args.connections
    )

    try:
        wf = client._open_wav()
        wf.close()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("=" * 60)
    print("Deepgram SageMaker WAV File Streaming Client")
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
    print(f"Limit:       {args.duration}s" if args.duration else "Limit:       until file ends" + (" (looping)" if args.loop else "") + " or Ctrl+C")
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
            keywords=keywords_list
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

    logger.info("Transcription complete!")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
