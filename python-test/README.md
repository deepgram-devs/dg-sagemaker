# Python Implementation

A Python project for testing Deepgram Text-to-Speech (TTS) using AWS SageMaker endpoints with bidirectional streaming. Supports multiple audio encoding formats (linear16, mulaw, alaw) with real-time audio playback.

## Prerequisites

- Python 3.8 or higher (tested with Python 3.13)
- pip package manager
- AWS credentials configured (via AWS CLI, environment variables, or IAM role)
- Access to a deployed SageMaker endpoint with Deepgram TTS model

## Installation

1. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Before running the scripts, you need to configure the following in `tts.py`:

- **Region**: Update the `region` variable (line 23) to match your SageMaker endpoint region
- **Endpoint Name**: Update the `endpointName` in the `config` dictionary (line 31) to your SageMaker endpoint name
- **Model**: The model is set to `aura-2-andromeda-en` by default. You can modify it in the `invoke_endpoint_with_bidirectional_stream()` function
- **Input Text**: Modify the `input_text` variable (line 29) to change the text that will be converted to speech

## Running the Scripts

### Run the main TTS script

Test a single voice with the default configuration (mulaw encoding):

```bash
python tts.py
```

Test with a specific encoding format:

```bash
# Use linear16 encoding (24kHz, 16-bit PCM)
python tts.py --encoding linear16

# Use mulaw encoding (8kHz, 8-bit mu-law)
python tts.py --encoding mulaw

# Use alaw encoding (8kHz, 8-bit A-law)
python tts.py --encoding alaw
```

### Run the test-all-voices script

Test all available Deepgram aura-2 English voices with default encoding (mulaw):

```bash
python test_all_voices.py
```

Test all voices with a specific encoding format:

```bash
# Test all voices with linear16 encoding
python test_all_voices.py --encoding linear16

# Test all voices with mulaw encoding
python test_all_voices.py --encoding mulaw

# Test all voices with alaw encoding
python test_all_voices.py --encoding alaw
```

### Command-Line Arguments

Both scripts support the following command-line argument:

- `--encoding`: Audio encoding format (choices: `linear16`, `mulaw`, `alaw`)
  - Default: `mulaw`
  - `linear16`: 16-bit, little-endian, signed PCM WAV data (typically 24kHz)
  - `mulaw`: Mu-law encoded WAV data (typically 8kHz)
  - `alaw`: A-law encoded WAV data (typically 8kHz)

## Scripts Overview

- **`tts.py`**: Main script that invokes a SageMaker endpoint with bidirectional streaming to generate and play TTS audio for a single voice. Supports multiple encoding formats with automatic decoding and playback.

- **`test_all_voices.py`**: Test script that iterates through all Deepgram aura-2 English voices and plays audio for each one. Can test all voices with a specified encoding format.

## Audio Encoding Support

The scripts support three audio encoding formats from Deepgram:

1. **linear16**: 16-bit, little-endian, signed PCM WAV data
   - Sample rate: Typically 24kHz
   - No decoding required (already PCM format)
   - Best quality, larger file size

2. **mulaw**: Mu-law encoded WAV data (ITU-T G.711)
   - Sample rate: Typically 8kHz
   - Automatically decoded to linear16 PCM for playback
   - Good quality, smaller file size
   - Default encoding

3. **alaw**: A-law encoded WAV data (ITU-T G.711)
   - Sample rate: Typically 8kHz
   - Automatically decoded to linear16 PCM for playback
   - Good quality, smaller file size
   - Alternative to mulaw (used in Europe)

The scripts automatically detect whether the audio data has a WAV header or is raw encoded data, and decode it appropriately before playback.

## Dependencies

- `boto3>=1.34.0` - AWS SDK for Python
- `sounddevice>=0.4.6` - Audio playback library for real-time audio output
- `numpy>=1.24.0` - Required for audio buffer handling and array operations
- `aws-sdk-sagemaker-runtime-http2>=0.1.0` - AWS SDK for SageMaker Runtime HTTP/2 client
- `smithy-aws-core>=0.2.0` - AWS SDK core library

All dependencies are pure Python or have Python 3.13 compatible implementations. No external audio codecs (like ffmpeg) are required.

## Features

- **Bidirectional Streaming**: Real-time text-to-speech with streaming audio responses
- **Multiple Encoding Support**: Supports linear16, mulaw, and alaw encoding formats
- **Automatic Decoding**: Automatically decodes mulaw and alaw to PCM for playback
- **Real-time Playback**: Audio is played directly through your default audio device
- **Command-line Interface**: Easy-to-use command-line arguments for encoding selection
- **Voice Testing**: Test individual voices or all available voices with a single command

## Technical Details

- **Text Chunking**: Text is split into 20-word chunks and sent with flush messages to optimize streaming
- **Metadata Handling**: The first chunk from SageMaker is a JSON metadata message that is automatically detected and skipped. Only audio chunks are processed.
- **Audio Decoding**: 
  - Mulaw and A-law use ITU-T G.711 standard decoding algorithms
  - Pure Python implementation (no external dependencies)
  - Automatically handles WAV headers or raw encoded data
  - Ensures proper byte alignment for linear16 encoding (16-bit samples)
- **Sample Rates**:
  - linear16: Typically 24kHz
  - mulaw/alaw: Typically 8kHz
- **Audio Format**: All formats are decoded/converted to 16-bit PCM (mono) for playback

## Notes

- The scripts use bidirectional streaming to send text chunks and receive audio responses in real-time
- Audio is played through your default audio device using the `sounddevice` package
- Make sure your AWS credentials are properly configured before running the scripts
- The scripts chunk text into 20-word segments and send them with flush messages to the SageMaker endpoint
- Audio encoding format can be specified via command-line argument (default: mulaw)
- The implementation uses pure Python for audio decoding, compatible with Python 3.13 (no audioop dependency)

