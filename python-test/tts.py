import asyncio
import json
import time
from typing import AsyncIterator, Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError
import sounddevice as sd
import numpy as np
from aws_sdk_sagemaker_runtime_http2.client import SageMakerRuntimeHTTP2Client
from aws_sdk_sagemaker_runtime_http2.config import Config, HTTPAuthSchemeResolver
from aws_sdk_sagemaker_runtime_http2.models import (
    InvokeEndpointWithBidirectionalStreamInput,
    RequestStreamEventPayloadPart,
    RequestPayloadPart
)
from smithy_aws_core.auth.sigv4 import SigV4AuthScheme
from smithy_aws_core.identity import EnvironmentCredentialsResolver
import os
import wave
import io
import argparse

# Configuration
region: str = "us-east-2"  # CHANGE ME: Specify the region where your SageMaker Endpoint is deployed.
bidi_endpoint: str = f"https://runtime.sagemaker.{region}.amazonaws.com:8443"  # CHANGE ME: This must correspond to the AWS region where your Endpoint is deployed.
model_invocation_path = 'v1/speak'  # The internal WebSocket API route you want to access, used by Deepgram specifically.
# model_invocation_path = '/invocations-bidirectional'  # Alternative path

input_text = """
The morning sun, filtered through the rustling leaves of the Sector 5 Slums. The beautiful trees swayed in the wind while Aeris watched them.
"""

config: Dict[str, str] = {
    'endpointName': 'deepgram-tts-test1',  # CHANGE ME: Update this value to the name of the SageMaker Endpoint you deploy
}

# Set up credentials from boto3 for EnvironmentCredentialsResolver
# This allows us to use boto3's credential chain (including ~/.aws/credentials)
# while still using the SDK's EnvironmentCredentialsResolver
def setup_credentials_from_boto3():
    """Set environment variables from boto3 credentials for SDK to use."""
    session = boto3.Session(region_name=region)
    credentials = session.get_credentials()
    if not credentials:
        raise Exception("No AWS credentials found. Please configure AWS credentials.")
    
    # Set environment variables that EnvironmentCredentialsResolver will read
    os.environ['AWS_ACCESS_KEY_ID'] = credentials.access_key
    os.environ['AWS_SECRET_ACCESS_KEY'] = credentials.secret_key
    if credentials.token:
        os.environ['AWS_SESSION_TOKEN'] = credentials.token
    elif 'AWS_SESSION_TOKEN' in os.environ:
        # Remove session token if it exists but credentials don't have one
        del os.environ['AWS_SESSION_TOKEN']

# Initialize credentials on module import
setup_credentials_from_boto3()


# Initialize SageMaker Runtime HTTP/2 client
def get_sagemaker_client() -> SageMakerRuntimeHTTP2Client:
    """Get configured SageMaker Runtime HTTP/2 client."""
    # Ensure credentials are set up
    setup_credentials_from_boto3()
    
    config = Config(
        endpoint_uri=bidi_endpoint,
        region=region,
        aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        auth_scheme_resolver=HTTPAuthSchemeResolver(),
        auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="sagemaker")}
    )
    return SageMakerRuntimeHTTP2Client(config=config)


def decode_mulaw(mulaw_data: bytes) -> bytes:
    """Decode mulaw-encoded audio to linear16 PCM using ITU-T G.711 standard."""
    linear16_data = bytearray(len(mulaw_data) * 2)
    
    for i, mulaw_byte in enumerate(mulaw_data):
        # Invert all bits (mulaw uses inverted encoding)
        mulaw_byte = mulaw_byte ^ 0xFF
        
        # Extract sign, exponent, and mantissa
        sign = (mulaw_byte & 0x80) >> 7
        exponent = (mulaw_byte & 0x70) >> 4
        mantissa = mulaw_byte & 0x0F
        
        # ITU-T G.711 mulaw decoding formula
        # linear = sign * ((33 + 2*mantissa) * 2^(exponent+2) - 33)
        linear = ((33 + 2 * mantissa) * (1 << (exponent + 2))) - 33
        
        # Apply sign
        if sign == 1:
            linear = -linear
        
        # Clamp to 16-bit signed integer range
        linear = max(-32768, min(32767, linear))
        
        # Convert to 16-bit signed integer (little-endian)
        linear16_data[i * 2] = linear & 0xFF
        linear16_data[i * 2 + 1] = (linear >> 8) & 0xFF
    
    return bytes(linear16_data)


def decode_alaw(alaw_data: bytes) -> bytes:
    """Decode A-law encoded audio to linear16 PCM using ITU-T G.711 standard."""
    linear16_data = bytearray(len(alaw_data) * 2)
    
    for i, alaw_byte in enumerate(alaw_data):
        # Invert even bits (A-law uses inverted encoding on even bits)
        alaw_byte = alaw_byte ^ 0x55
        
        # Extract sign, exponent, and mantissa
        sign = (alaw_byte & 0x80) >> 7
        exponent = (alaw_byte & 0x70) >> 4
        mantissa = alaw_byte & 0x0F
        
        # ITU-T G.711 A-law decoding formula
        if exponent == 0:
            # Special case for exponent 0
            linear = 2 * mantissa + 33
        else:
            # Standard case
            linear = ((2 * mantissa + 33) * (1 << (exponent + 1))) - 33
        
        # Apply sign
        if sign == 1:
            linear = -linear
        
        # Clamp to 16-bit signed integer range
        linear = max(-32768, min(32767, linear))
        
        # Convert to 16-bit signed integer (little-endian)
        linear16_data[i * 2] = linear & 0xFF
        linear16_data[i * 2 + 1] = (linear >> 8) & 0xFF
    
    return bytes(linear16_data)


async def sleep(ms: int) -> None:
    """Sleep for specified milliseconds."""
    await asyncio.sleep(ms / 1000.0)
    print("Sleeping completed!")


async def create_tts_request_events():
    """Create an async generator that yields TTS request events for AWS SDK."""
    # Split the input text into chunks of 20 words each
    words = input_text.strip().split()
    chunk_size = 20
    chunks: list[str] = []
    
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    print(f"Splitting text into {len(chunks)} chunks of ~{chunk_size} words each")
    
    # Yield each chunk as a RequestStreamEventPayloadPart
    for i, chunk in enumerate(chunks):
        tts_payload = {
            "type": "Speak",
            "text": chunk
        }
        
        if chunk:
            preview = chunk[:50] + "..." if len(chunk) > 50 else chunk
            print(f'Sending chunk {i + 1}/{len(chunks)}: "{preview}"')
        
        # Create payload part
        payload_bytes = json.dumps(tts_payload).encode('utf-8')
        # Try adding data_type parameter to match JavaScript DataType: "UTF8"
        try:
            payload_part = RequestPayloadPart(bytes_=payload_bytes, data_type="UTF8")
        except TypeError:
            # If data_type parameter doesn't exist, try dataType
            try:
                payload_part = RequestPayloadPart(bytes_=payload_bytes, dataType="UTF8")
            except TypeError:
                # Fall back to just bytes_ if neither works
                payload_part = RequestPayloadPart(bytes_=payload_bytes)
        event = RequestStreamEventPayloadPart(value=payload_part)
        yield event
        
        await sleep(3500)
        print('Sending flush message')
        flush_payload = json.dumps({"type": "Flush"}).encode('utf-8')
        try:
            flush_part = RequestPayloadPart(bytes_=flush_payload, data_type="UTF8")
        except TypeError:
            try:
                flush_part = RequestPayloadPart(bytes_=flush_payload, dataType="UTF8")
            except TypeError:
                flush_part = RequestPayloadPart(bytes_=flush_payload)
        flush_event = RequestStreamEventPayloadPart(value=flush_part)
        yield flush_event
    
    # Sleep to ensure the complete response stream is received
    await sleep(5000)


async def invoke_endpoint_with_bidirectional_stream(encoding: str = 'mulaw') -> Dict[str, Any]:
    """Invoke SageMaker endpoint with bidirectional streaming using AWS SDK."""
    print('Invoking endpoint with bidirectional stream...')
    
    try:
        print(f'Using custom bidi endpoint: {bidi_endpoint}')
        print('Sending bidirectional stream request...')
        
        # Get the SageMaker client
        client = get_sagemaker_client()
        
        # Build query string with encoding parameter
        model_query_string = f'model=aura-2-andromeda-en&encoding={encoding}'
        
        # Create the input with model invocation path and query string
        input_params = InvokeEndpointWithBidirectionalStreamInput(
            endpoint_name=config['endpointName'],
            model_invocation_path=model_invocation_path,
            model_query_string=model_query_string
        )
        
        # Invoke the endpoint with bidirectional streaming
        stream = await client.invoke_endpoint_with_bidirectional_stream(input_params)
        
        print('Bidirectional stream established')
        
        # Send request events
        async def send_events():
            async for event in create_tts_request_events():
                await stream.input_stream.send(event)
            await stream.input_stream.close()
            print('Input stream closed')
        
        # Start sending events in background
        send_task = asyncio.create_task(send_events())
        
        # Process responses
        audio_chunks: list[bytes] = []
        chunk_count = 0
        
        try:
            output = await stream.await_output()
            output_stream = output[1]
            
            try:
                await asyncio.wait_for(send_task, timeout=60.0)
            except asyncio.TimeoutError:
                pass
            timeout_seconds = 30
            start_time = time.time()
            consecutive_none_count = 0
            max_consecutive_none = 5  # Allow more None results before giving up
            last_chunk_time = time.time()
            
            while True:
                elapsed = time.time() - start_time
                time_since_last_chunk = time.time() - last_chunk_time
                
                if send_task.done() and time_since_last_chunk > 10.0 and chunk_count > 0:
                    break
                
                if elapsed > timeout_seconds:
                    break
                
                try:
                    result = await asyncio.wait_for(output_stream.receive(), timeout=5.0)
                except asyncio.TimeoutError:
                    if send_task.done() and chunk_count > 0:
                        await asyncio.sleep(2.0)
                        try:
                            result = await asyncio.wait_for(output_stream.receive(), timeout=1.0)
                        except asyncio.TimeoutError:
                            break
                    else:
                        continue
                
                if result is None:
                    consecutive_none_count += 1
                    
                    if consecutive_none_count >= max_consecutive_none and send_task.done():
                        await asyncio.sleep(2.0)
                        try:
                            result = await asyncio.wait_for(output_stream.receive(), timeout=1.0)
                            if result is not None:
                                consecutive_none_count = 0
                                continue
                        except asyncio.TimeoutError:
                            pass
                        break
                    else:
                        await asyncio.sleep(0.5)
                        continue
                
                consecutive_none_count = 0
                chunk_count += 1
                last_chunk_time = time.time()
                
                # Check for stream errors first (similar to JavaScript)
                if hasattr(result, 'internal_stream_failure') and result.internal_stream_failure:
                    print(f'Internal stream failure: {result.internal_stream_failure}')
                    break
                
                if hasattr(result, 'model_stream_error') and result.model_stream_error:
                    print(f'Model stream error: {result.model_stream_error}')
                    break
                
                audio_data = None
                
                if hasattr(result, 'payload_part') and result.payload_part:
                    payload_part = result.payload_part
                    if hasattr(payload_part, 'bytes_'):
                        audio_data = payload_part.bytes_
                    elif hasattr(payload_part, 'bytes'):
                        audio_data = payload_part.bytes
                
                if not audio_data and hasattr(result, 'value') and result.value:
                    if hasattr(result.value, 'bytes_'):
                        audio_data = result.value.bytes_
                    elif hasattr(result.value, 'bytes'):
                        audio_data = result.value.bytes
                    elif hasattr(result.value, 'payload_part') and result.value.payload_part:
                        payload_part = result.value.payload_part
                        if hasattr(payload_part, 'bytes_'):
                            audio_data = payload_part.bytes_
                        elif hasattr(payload_part, 'bytes'):
                            audio_data = payload_part.bytes
                
                if not audio_data:
                    if hasattr(result, 'bytes_'):
                        audio_data = result.bytes_
                    elif hasattr(result, 'bytes'):
                        audio_data = result.bytes
                
                if audio_data:
                    is_metadata = audio_data.startswith(b'{"type":"Metadata"') or audio_data.startswith(b'{"type": "Metadata"')
                    if not is_metadata:
                        audio_chunks.append(audio_data)
        except Exception as e:
            print(f'Error receiving responses: {e}')
            import traceback
            traceback.print_exc()
        
        # Ensure send task is complete 
        if not send_task.done():
            await send_task
        
        response = {'Body': b''.join(audio_chunks) if audio_chunks else b''}
        response_body = response['Body']
        
        if len(response_body) > 0:
            if len(response_body) < 1000:
                print(f'Warning: Very small audio buffer ({len(response_body)} bytes). Audio may be inaudible.')
            
            if len(response_body) > 0:
                
                if response_body[:4] == b'RIFF':
                    wav_file = wave.open(io.BytesIO(response_body), 'rb')
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    num_frames = wav_file.getnframes()
                    encoded_data = wav_file.readframes(num_frames)
                    wav_file.close()
                else:
                    encoded_data = response_body
                    sample_rate = 8000 if encoding in ['mulaw', 'alaw'] else 24000
                
                if encoding == 'linear16':
                    linear16_data = encoded_data
                elif encoding == 'mulaw':
                    # Decode mulaw to linear16 PCM
                    linear16_data = decode_mulaw(encoded_data)
                elif encoding == 'alaw':
                    # Decode A-law to linear16 PCM
                    linear16_data = decode_alaw(encoded_data)
                else:
                    raise ValueError(f'Unsupported encoding format: {encoding}')
                
                if len(linear16_data) % 2 != 0:
                    linear16_data = linear16_data[:-1]
                
                # Convert to numpy array (16-bit samples)
                audio_array = np.frombuffer(linear16_data, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0
                if len(audio_float.shape) > 1:
                    audio_float = audio_float.flatten()
                
                sd.play(audio_float, samplerate=sample_rate)
                sd.wait()
        return response
        
    except Exception as error:
        print(f'Error invoking endpoint with bidirectional stream: {error}')
        import traceback
        traceback.print_exc()
        raise error


# Main execution function
async def main(encoding: str = 'mulaw') -> None:
    """Main execution function."""
    try:
        print(f'Starting SageMaker deployment process with encoding: {encoding}...')
        
        await invoke_endpoint_with_bidirectional_stream(encoding=encoding)
        
        print('All operations completed successfully!')
    
    except Exception as error:
        print(f'Deployment process failed: {error}')
        raise error


# Run the script if this file is executed directly
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deepgram TTS via SageMaker')
    parser.add_argument(
        '--encoding',
        type=str,
        choices=['linear16', 'mulaw', 'alaw'],
        default='mulaw',
        help='Audio encoding format: linear16, mulaw, or alaw (default: mulaw)'
    )
    args = parser.parse_args()
    
    try:
        asyncio.run(main(encoding=args.encoding))
    except Exception as error:
        print(f'Script execution failed: {error}')
        exit(1)

