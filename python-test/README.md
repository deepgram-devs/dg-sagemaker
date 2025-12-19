# Python Implementation

A Python project for testing Deepgram Text-to-Speech (TTS) using AWS SageMaker endpoints with bidirectional streaming.

## Prerequisites

- Python 3.8 or higher
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

- **Region**: Update the `region` variable (line 17) to match your SageMaker endpoint region
- **Endpoint Name**: Update the `endpointName` in the `config` dictionary (line 29) to your SageMaker endpoint name
- **Model**: Update the `model_query_string` (line 21) to specify the Deepgram voice model you want to use
- **Input Text**: Modify the `input_text` variable (line 23) to change the text that will be converted to speech

## Running the Scripts

### Run the main TTS script

Test a single voice with the default configuration:

```bash
python tts.py
```

### Run the test-all-voices script

Test all available Deepgram aura-2 English voices:

```bash
python test_all_voices.py
```

## Scripts Overview

- **`tts.py`**: Main script that invokes a SageMaker endpoint with bidirectional streaming to generate and play TTS audio for a single voice
- **`test_all_voices.py`**: Test script that iterates through all Deepgram aura-2 English voices and plays audio for each one

## Dependencies

- `boto3>=1.34.0` - AWS SDK for Python
- `sounddevice>=0.4.6` - Audio playback library
- `numpy>=1.24.0` - Required by sounddevice for audio buffer handling

## Notes

- The scripts use bidirectional streaming to send text chunks and receive audio responses in real-time
- Audio is played through your default audio device using the `sounddevice` package
- Make sure your AWS credentials are properly configured before running the scripts
- The scripts chunk text into 20-word segments and send them with flush messages to the SageMaker endpoint
- Audio is received in linear16 format at 24kHz sample rate (mono, 16-bit)

