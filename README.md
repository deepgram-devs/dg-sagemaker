# aws-deepgram-tts-sagemaker

A project for testing Deepgram Text-to-Speech (TTS) using AWS SageMaker endpoints with bidirectional streaming. Available in both JavaScript/TypeScript and Python implementations.

## Project Structure

- `js-test/` - JavaScript/TypeScript implementation (see [js-test/README.md](js-test/README.md))
- `python-test/` - Python implementation (see [python-test/README.md](python-test/README.md))

## Prerequisites

- AWS credentials configured (via AWS CLI, environment variables, or IAM role)
- Access to a deployed SageMaker endpoint with Deepgram TTS model

## Quick Start

### JavaScript/TypeScript

See [js-test/README.md](js-test/README.md) for detailed instructions.

```bash
cd js-test
yarn install
yarn tsx tts.ts
```

### Python

See [python-test/README.md](python-test/README.md) for detailed instructions.

```bash
cd python-test
pip install -r requirements.txt
python tts.py
```

## Notes

- Both implementations use bidirectional streaming to send text chunks and receive audio responses in real-time
- Audio is played through your default audio device
- The scripts chunk text into 20-word segments and send them with flush messages to the SageMaker endpoint
- Audio is received in linear16 format at 24kHz sample rate (mono, 16-bit)
