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
from tts import config, region, model_invocation_path, get_sagemaker_client
import os
import wave
import io

# All English Deepgram aura-2 voices
english_aura2_voices = [
    'aura-2-amalthea-en',
    'aura-2-andromeda-en',
    'aura-2-apollo-en',
    'aura-2-arcas-en',
    'aura-2-aries-en',
    'aura-2-asteria-en',
    'aura-2-athena-en',
    'aura-2-atlas-en',
    'aura-2-aurora-en',
    'aura-2-callista-en',
    'aura-2-cora-en',
    'aura-2-cordelia-en',
    'aura-2-delia-en',
    'aura-2-draco-en',
    'aura-2-electra-en',
    'aura-2-harmonia-en',
    'aura-2-helena-en',
    'aura-2-hera-en',
    'aura-2-hermes-en',
    'aura-2-hyperion-en',
    'aura-2-iris-en',
    'aura-2-janus-en',
    'aura-2-juno-en',
    'aura-2-jupiter-en',
    'aura-2-luna-en',
    'aura-2-mars-en',
    'aura-2-minerva-en',
    'aura-2-neptune-en',
    'aura-2-odysseus-en',
    'aura-2-ophelia-en',
    'aura-2-orion-en',
    'aura-2-orpheus-en',
    'aura-2-pandora-en',
    'aura-2-phoebe-en',
    'aura-2-pluto-en',
    'aura-2-saturn-en',
    'aura-2-selene-en',
    'aura-2-thalia-en',
    'aura-2-theia-en',
    'aura-2-vesta-en',
    'aura-2-zeus-en',
]

# Configuration from tts.py
input_text = """
The morning sun, filtered through the rustling leaves of the Sector 5 Slums. The beautiful trees swayed in the wind while Aeris watched them.
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

    await sleep(5000)


async def sleep(ms: int) -> None:
    """Sleep for specified milliseconds."""
    await asyncio.sleep(ms / 1000.0)
    print("Sleeping completed!")




async def invoke_endpoint_with_voice(voice_id: str) -> Dict[str, Any]:
    """Invoke SageMaker endpoint with a specific voice using AWS SDK."""
    print(f"\n{'=' * 60}")
    print(f"Testing voice: {voice_id}")
    print('=' * 60)

    model_query_string = f'model={voice_id}&encoding=mulaw'

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
            
            # First, wait for send task to complete (or timeout)
            print('Waiting for send task to complete...')
            try:
                await asyncio.wait_for(send_task, timeout=60.0)
                print('Send task completed')
            except asyncio.TimeoutError:
                print('Warning: Send task timed out, continuing to receive responses')
            
            # Now receive all responses
            # Wait for responses with timeout after send is done
            timeout_seconds = 30
            start_time = time.time()
            consecutive_none_count = 0
            max_consecutive_none = 5  # Allow more None results before giving up
            last_chunk_time = time.time()
            
            while True:
                # Check timeout - but only if we haven't received anything recently
                elapsed = time.time() - start_time
                time_since_last_chunk = time.time() - last_chunk_time
                
                # If send is done and we haven't received anything for 10 seconds, we're probably done
                if send_task.done() and time_since_last_chunk > 10.0 and chunk_count > 0:
                    print(f'No more responses (no chunks for {time_since_last_chunk:.1f}s after send complete)')
                    break
                
                # Overall timeout
                if elapsed > timeout_seconds:
                    print(f'Overall timeout ({timeout_seconds}s) waiting for responses')
                    break
                
                try:
                    # Use asyncio.wait_for to add timeout to receive
                    result = await asyncio.wait_for(output_stream.receive(), timeout=5.0)
                except asyncio.TimeoutError:
                    # If send is done and we've received chunks, might be finished
                    if send_task.done() and chunk_count > 0:
                        # Wait a bit more for any final chunks
                        await asyncio.sleep(2.0)
                        # Try one more time
                        try:
                            result = await asyncio.wait_for(output_stream.receive(), timeout=1.0)
                        except asyncio.TimeoutError:
                            print('No more responses (timeout after send complete)')
                            break
                    else:
                        continue
                
                if result is None:
                    # None doesn't necessarily mean the stream is done
                    # Continue trying to receive, but track consecutive None results
                    consecutive_none_count += 1
                    print(f'Received None (count: {consecutive_none_count}), continuing to wait for more chunks...')
                    
                    # Only break if we get many None results AND send is done AND we've waited
                    if consecutive_none_count >= max_consecutive_none and send_task.done():
                        # Wait a bit more to see if more chunks arrive
                        await asyncio.sleep(2.0)
                        # Try one more receive
                        try:
                            result = await asyncio.wait_for(output_stream.receive(), timeout=1.0)
                            if result is not None:
                                consecutive_none_count = 0
                                # Process this result in the next iteration
                                continue
                        except asyncio.TimeoutError:
                            pass
                        
                        print(f'No more responses after {consecutive_none_count} None results (send complete)')
                        break
                    else:
                        # Wait a bit and try again
                        await asyncio.sleep(0.5)
                        continue
                
                # Reset None counter when we get a valid result
                consecutive_none_count = 0
                chunk_count += 1
                last_chunk_time = time.time()  # Update last chunk time
                
                # Debug: Print result type and attributes
                result_type = type(result).__name__
                attrs = [attr for attr in dir(result) if not attr.startswith('_')]
                print(f'Processing chunk {chunk_count}, type: {result_type}, attributes: {attrs[:15]}')
                
                # Check for stream errors first (similar to JavaScript)
                if hasattr(result, 'internal_stream_failure') and result.internal_stream_failure:
                    print(f'Internal stream failure: {result.internal_stream_failure}')
                    break
                
                if hasattr(result, 'model_stream_error') and result.model_stream_error:
                    print(f'Model stream error: {result.model_stream_error}')
                    break
                
                # Extract audio data from PayloadPart (similar to JavaScript chunk.PayloadPart.Bytes)
                audio_data = None
                
                # Check for payload_part attribute (similar to JavaScript chunk.PayloadPart)
                if hasattr(result, 'payload_part') and result.payload_part:
                    payload_part = result.payload_part
                    print(f'  Found payload_part: {type(payload_part).__name__}')
                    if hasattr(payload_part, 'bytes_'):
                        audio_data = payload_part.bytes_
                        print(f'  Extracted audio from payload_part.bytes_: {len(audio_data) if audio_data else 0} bytes')
                    elif hasattr(payload_part, 'bytes'):
                        audio_data = payload_part.bytes
                        print(f'  Extracted audio from payload_part.bytes: {len(audio_data) if audio_data else 0} bytes')
                
                # Fallback to checking value attribute
                if not audio_data and hasattr(result, 'value') and result.value:
                    print(f'  Checking result.value: {type(result.value).__name__}')
                    if hasattr(result.value, 'bytes_'):
                        audio_data = result.value.bytes_
                        print(f'  Extracted audio from value.bytes_: {len(audio_data) if audio_data else 0} bytes')
                    elif hasattr(result.value, 'bytes'):
                        audio_data = result.value.bytes
                        print(f'  Extracted audio from value.bytes: {len(audio_data) if audio_data else 0} bytes')
                    # Also check if value has payload_part
                    elif hasattr(result.value, 'payload_part') and result.value.payload_part:
                        payload_part = result.value.payload_part
                        print(f'  Found value.payload_part: {type(payload_part).__name__}')
                        if hasattr(payload_part, 'bytes_'):
                            audio_data = payload_part.bytes_
                            print(f'  Extracted audio from value.payload_part.bytes_: {len(audio_data) if audio_data else 0} bytes')
                        elif hasattr(payload_part, 'bytes'):
                            audio_data = payload_part.bytes
                            print(f'  Extracted audio from value.payload_part.bytes: {len(audio_data) if audio_data else 0} bytes')
                
                # Final fallback
                if not audio_data:
                    if hasattr(result, 'bytes_'):
                        audio_data = result.bytes_
                        print(f'  Extracted audio from result.bytes_: {len(audio_data) if audio_data else 0} bytes')
                    elif hasattr(result, 'bytes'):
                        audio_data = result.bytes
                        print(f'  Extracted audio from result.bytes: {len(audio_data) if audio_data else 0} bytes')
                
                if audio_data:
                    if len(audio_data) > 34:  # Skip small metadata messages
                        audio_chunks.append(audio_data)
                        print(f'✓ Received audio chunk {chunk_count}, size: {len(audio_data)} bytes')
                    else:
                        print(f'⊘ Skipping small chunk {chunk_count}, size: {len(audio_data)} bytes (likely metadata)')
                else:
                    # Log result structure for debugging
                    print(f'✗ Chunk {chunk_count} has no audio data. Full attributes: {attrs}')
                    # Try to print the actual result object
                    try:
                        print(f'  Result repr: {repr(result)[:200]}')
                    except:
                        pass
        except Exception as e:
            print(f'Error receiving responses: {e}')
            import traceback
            traceback.print_exc()
        
        # Ensure send task is complete (should already be done, but just in case)
        if not send_task.done():
            await send_task
        
        print(f'Received {chunk_count} chunks, {len(audio_chunks)} audio chunks')

        if audio_chunks:
            print(f'Playing {len(audio_chunks)} audio chunks for voice {voice_id}...')
            audio_buffer = b''.join(audio_chunks)
            print(f'Total audio buffer size: {len(audio_buffer)} bytes')
            
            # Check if we have enough audio data
            if len(audio_buffer) < 1000:
                print(f'Warning: Very small audio buffer ({len(audio_buffer)} bytes). Audio may be inaudible.')
            
            if len(audio_buffer) == 0:
                print('No audio data to play')
            else:
                # Decode mulaw audio and play
                # Check if data starts with RIFF (WAV header) or is raw mulaw
                if audio_buffer[:4] == b'RIFF':
                    # It's a WAV file, read the header
                    wav_file = wave.open(io.BytesIO(audio_buffer), 'rb')
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    num_frames = wav_file.getnframes()
                    mulaw_data = wav_file.readframes(num_frames)
                    wav_file.close()
                else:
                    # Raw mulaw data (no WAV header)
                    mulaw_data = audio_buffer
                    # Mulaw is typically 8000 Hz, mono
                    sample_rate = 8000
                
                # Decode mulaw to linear16 PCM using ITU-T G.711 standard
                # Mulaw decoding: convert 8-bit mulaw to 16-bit linear PCM
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
                
                # Convert to numpy array (16-bit samples)
                audio_array = np.frombuffer(bytes(linear16_data), dtype=np.int16)
                print(f'Audio array shape: {audio_array.shape}, samples: {len(audio_array)}')
                
                # Normalize to float32 range [-1.0, 1.0] for sounddevice
                audio_float = audio_array.astype(np.float32) / 32768.0
                
                # Reshape for mono audio
                if len(audio_float.shape) > 1:
                    audio_float = audio_float.flatten()
                
                # Play the audio
                duration = len(audio_float) / sample_rate
                print(f'Playing audio: {len(audio_float)} samples at {sample_rate} Hz (duration: {duration:.2f} seconds)')
                sd.play(audio_float, samplerate=sample_rate)
                sd.wait()
                print(f'Audio playback completed for {voice_id}')
        else:
            print(f'No audio chunks received for voice {voice_id}')
        
        # Return response dict for compatibility
        response = {'Body': b''.join(audio_chunks) if audio_chunks else b''}

        print(f'Voice {voice_id} completed successfully')
        return response

    except Exception as error:
        print(f'Error invoking endpoint with voice {voice_id}: {error}')
        import traceback
        traceback.print_exc()
        raise error


async def test_all_voices() -> None:
    """Test all available Deepgram aura-2 English voices."""
    print(f'Starting test of {len(english_aura2_voices)} Deepgram aura-2 English voices...\n')

    results: List[Dict[str, Any]] = []

    for voice in english_aura2_voices:
        try:
            await invoke_endpoint_with_voice(voice)
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
async def main() -> None:
    """Main execution function."""
    try:
        await test_all_voices()
        print('\nAll voice tests completed!')
    except Exception as error:
        print(f'Voice testing process failed: {error}')
        raise error


# Run the script if this file is executed directly
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as error:
        print(f'Script execution failed: {error}')
        exit(1)

