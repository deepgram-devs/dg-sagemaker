import asyncio
import json
import time
from typing import Dict, Any, List
import sounddevice as sd
import numpy as np
from aws_sdk_sagemaker_runtime_http2.models import (
    InvokeEndpointWithBidirectionalStreamInput,
    RequestStreamEventPayloadPart,
    RequestPayloadPart
)
from tts import config, region, model_invocation_path, get_sagemaker_client, decode_mulaw, decode_alaw
import os
import wave
import io
import argparse

# All English Deepgram aura-2 voices
english_aura2_voices = [
    'aura-2-amalthea-en',
    # 'aura-2-andromeda-en',
    'aura-2-apollo-en',
    'aura-2-arcas-en',
    'aura-2-aries-en',
    # 'aura-2-asteria-en',
    # 'aura-2-athena-en',
    # 'aura-2-atlas-en',
    # 'aura-2-aurora-en',
    # 'aura-2-callista-en',
    # 'aura-2-cora-en',
    # 'aura-2-cordelia-en',
    # 'aura-2-delia-en',
    # 'aura-2-draco-en',
    # 'aura-2-electra-en',
    # 'aura-2-harmonia-en',
    # 'aura-2-helena-en',
    # 'aura-2-hera-en',
    # 'aura-2-hermes-en',
    # 'aura-2-hyperion-en',
    # 'aura-2-iris-en',
    # 'aura-2-janus-en',
    # 'aura-2-juno-en',
    # 'aura-2-jupiter-en',
    # 'aura-2-luna-en',
    # 'aura-2-mars-en',
    # 'aura-2-minerva-en',
    # 'aura-2-neptune-en',
    # 'aura-2-odysseus-en',
    # 'aura-2-ophelia-en',
    # 'aura-2-orion-en',
    # 'aura-2-orpheus-en',
    # 'aura-2-pandora-en',
    # 'aura-2-phoebe-en',
    # 'aura-2-pluto-en',
    # 'aura-2-saturn-en',
    # 'aura-2-selene-en',
    # 'aura-2-thalia-en',
    # 'aura-2-theia-en',
    # 'aura-2-vesta-en',
    # 'aura-2-zeus-en',
]

# Configuration from tts.py
input_text = """
The morning sun, filtered through the rustling leaves of the Sector 5 Slums.
"""

async def create_tts_request_events():
    """Create an async generator that yields TTS request events for AWS SDK."""
    words = input_text.strip().split()
    chunk_size = 20
    chunks: List[str] = []

    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)

    print(f"Splitting text into {len(chunks)} chunks of ~{chunk_size} words each")

    for i, chunk in enumerate(chunks):
        tts_payload = {
            "type": "Speak",
            "text": chunk
        }

        if chunk:
            preview = chunk[:50] + "..." if len(chunk) > 50 else chunk
            print(f'Sending chunk {i + 1}/{len(chunks)}: "{preview}"')

        payload_bytes = json.dumps(tts_payload).encode('utf-8')
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

    await sleep(5000)


async def sleep(ms: int) -> None:
    """Sleep for specified milliseconds."""
    await asyncio.sleep(ms / 1000.0)
    print("Sleeping completed!")




async def invoke_endpoint_with_voice(voice_id: str, encoding: str = 'mulaw') -> Dict[str, Any]:
    """Invoke SageMaker endpoint with a specific voice using AWS SDK."""
    print(f"\n{'=' * 60}")
    print(f"Testing voice: {voice_id} with encoding: {encoding}")
    print('=' * 60)

    model_query_string = f'model={voice_id}&encoding={encoding}'

    try:
        print('Sending bidirectional stream request...')

        # Get the SageMaker client
        client = get_sagemaker_client()
        
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
        audio_chunks: List[bytes] = []
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
                        if encoding == 'linear16' and len(audio_data) % 2 != 0:
                            audio_data = audio_data[:-1]
                        audio_chunks.append(audio_data)
        except Exception as e:
            print(f'Error receiving responses: {e}')
            import traceback
            traceback.print_exc()
        
        if not send_task.done():
            await send_task

        if audio_chunks:
            audio_buffer = b''.join(audio_chunks)
            
            if len(audio_buffer) < 1000:
                print(f'Warning: Very small audio buffer ({len(audio_buffer)} bytes). Audio may be inaudible.')
            
            if len(audio_buffer) > 0:
                if audio_buffer[:4] == b'RIFF':
                    wav_file = wave.open(io.BytesIO(audio_buffer), 'rb')
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    num_frames = wav_file.getnframes()
                    encoded_data = wav_file.readframes(num_frames)
                    wav_file.close()
                else:
                    encoded_data = audio_buffer
                    sample_rate = 8000 if encoding in ['mulaw', 'alaw'] else 24000
                
                if encoding == 'linear16':
                    linear16_data = encoded_data
                elif encoding == 'mulaw':
                    linear16_data = decode_mulaw(encoded_data)
                elif encoding == 'alaw':
                    linear16_data = decode_alaw(encoded_data)
                else:
                    raise ValueError(f'Unsupported encoding format: {encoding}')
                
                if len(linear16_data) % 2 != 0:
                    linear16_data = linear16_data[:-1]
                
                audio_array = np.frombuffer(linear16_data, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0
                if len(audio_float.shape) > 1:
                    audio_float = audio_float.flatten()
                
                sd.play(audio_float, samplerate=sample_rate)
                sd.wait()
        
        response = {'Body': b''.join(audio_chunks) if audio_chunks else b''}
        return response

    except Exception as error:
        print(f'Error invoking endpoint with voice {voice_id}: {error}')
        import traceback
        traceback.print_exc()
        raise error


async def test_all_voices(encoding: str = 'mulaw') -> None:
    """Test all available Deepgram aura-2 English voices."""
    print(f'Starting test of {len(english_aura2_voices)} Deepgram aura-2 English voices with encoding: {encoding}...\n')

    results: List[Dict[str, Any]] = []

    for voice in english_aura2_voices:
        try:
            await invoke_endpoint_with_voice(voice, encoding=encoding)
            results.append({'voice': voice, 'success': True})
            print(f'✓ {voice} - Success\n')
        except Exception as error:
            error_message = str(error) if error else 'Unknown error'
            results.append({
                'voice': voice,
                'success': False,
                'error': error_message
            })
            print(f'✗ {voice} - Failed: {error_message}\n')

        # Add a small delay between voices to avoid overwhelming the endpoint
        await sleep(2000)

    # Print summary
    print('\n' + '=' * 60)
    print('TEST SUMMARY')
    print('=' * 60)
    print(f'Total voices tested: {len(results)}')
    print(f'Successful: {len([r for r in results if r["success"]])}')
    print(f'Failed: {len([r for r in results if not r["success"]])}')

    failures = [r for r in results if not r['success']]
    if failures:
        print('\nFailed voices:')
        for f in failures:
            print(f'  - {f["voice"]}: {f.get("error", "Unknown error")}')


# Main execution
async def main(encoding: str = 'mulaw') -> None:
    """Main execution function."""
    try:
        await test_all_voices(encoding=encoding)
        print('\nAll voice tests completed!')
    except Exception as error:
        print(f'Voice testing process failed: {error}')
        raise error


# Run the script if this file is executed directly
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test all Deepgram TTS voices via SageMaker')
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

