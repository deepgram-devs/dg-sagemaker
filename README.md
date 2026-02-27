# aws-deepgram-sagemaker

## Test Deepgram Transcription on SageMaker

### JavaScript


### Python

Follow these steps if you'd like to test the Deepgram Speech-to-Text (STT) Amazon SageMaker

- Ensure you've deployed an Amazon SageMaker endpoint
- Install the `uv` [package manager](https://github.com/astral-sh/uv)
- Install dependencies: `uv pip install -r requirements.txt`

#### Python Microphone Stress Test Examples

The `stt_microphone_stress.py` script streams live microphone audio to Deepgram on SageMaker for real-time transcription. It supports multiple simultaneous connections for load testing.

**Basic usage (single connection):**

```bash
uv run python-stt/stt_microphone_stress.py your-endpoint-name
```

**With specific AWS region:**

```bash
uv run python-stt/stt_microphone_stress.py your-endpoint-name --region us-west-2
```

**Multiple simultaneous connections (load testing):**

```bash
uv run python-stt/stt_microphone_stress.py your-endpoint-name --connections 5
```

**With speaker diarization:**

```bash
uv run python-stt/stt_microphone_stress.py your-endpoint-name --diarize true
```

**With different model and language:**

The default Speech-to-Text (STT) transcription model is `nova-3`.

```bash
uv run python-stt/stt_microphone_stress.py your-endpoint-name --model nova-2 --language es
```

**With keywords boosting:**

Keywords are only conmpatible with `nova-2`. For `nova-3` use keyterms instead.
```bash
uv run python-stt/stt_microphone_stress.py your-endpoint-name --keywords "Deepgram:5,SageMaker:10,transcription:3"
```

**Run for a fixed duration (useful for automated tests):**

```bash
uv run python-stt/stt_microphone_stress.py your-endpoint-name --duration 30
```

**Timed load test with multiple connections:**

```bash
uv run python-stt/stt_microphone_stress.py your-endpoint-name --connections 5 --duration 120
```

**Full example with all options:**

```bash
uv run python-stt/stt_microphone_stress.py your-endpoint-name \
  --connections 3 \
  --model nova-2 \
  --language en \
  --diarize true \
  --punctuate true \
  --keywords "hello:5,world:10" \
  --duration 60 \
  --region us-east-1 \
  --log-level DEBUG
```

**Available options:**

- `--connections N` - Number of simultaneous streaming connections (default: 1)
- `--model MODEL` - Deepgram model to use (default: nova-3)
- `--language LANG` - Language code (default: en)
- `--diarize true|false` - Enable speaker diarization (default: false)
- `--punctuate true|false` - Enable punctuation (default: true)
- `--keywords KEYWORDS` - Comma-delimited keywords with intensity (format: "word:intensity,word:intensity")
- `--duration SECONDS` - Stop automatically after this many seconds (default: run until Ctrl+C)
- `--region REGION` - AWS region (default: us-east-1)
- `--log-level LEVEL` - Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)

#### Python WAV File Stress Test Examples

The `stt_wav_stress.py` script streams a WAV audio file to Deepgram on SageMaker in real-time, paced to match the file's actual sample rate. This makes it behave like a live microphone source, enabling repeatable and automated load testing without requiring a physical microphone.

> **Requirements:** The WAV file must be 16-bit PCM (linear16). To convert any audio file, use `ffmpeg`:
> ```bash
> ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 output.wav
> ```

**Basic usage (single connection, play file once):**

```bash
uv run python-stt/stt_wav_stress.py your-endpoint-name --file audio.wav
```

**With a specific AWS region:**

```bash
uv run python-stt/stt_wav_stress.py your-endpoint-name --file audio.wav --region us-west-2
```

**Multiple simultaneous connections (load testing):**

```bash
uv run python-stt/stt_wav_stress.py your-endpoint-name --file audio.wav --connections 5
```

**Loop the file continuously until Ctrl+C:**

```bash
uv run python-stt/stt_wav_stress.py your-endpoint-name --file audio.wav --loop
```

**Loop for a fixed duration (useful for automated tests):**

```bash
uv run python-stt/stt_wav_stress.py your-endpoint-name --file audio.wav --loop --duration 120
```

**Timed load test with multiple connections:**

```bash
uv run python-stt/stt_wav_stress.py your-endpoint-name --file audio.wav --connections 10 --loop --duration 300
```

**With speaker diarization:**

```bash
uv run python-stt/stt_wav_stress.py your-endpoint-name --file audio.wav --diarize true
```

**With a different model and language:**

```bash
uv run python-stt/stt_wav_stress.py your-endpoint-name --file audio.wav --model nova-2 --language es
```

**With keywords boosting:**

Keywords are only compatible with `nova-2`. For `nova-3` use keyterms instead.

```bash
uv run python-stt/stt_wav_stress.py your-endpoint-name --file audio.wav --keywords "Deepgram:5,SageMaker:10"
```

**Full example with all options:**

```bash
uv run python-stt/stt_wav_stress.py your-endpoint-name --file audio.wav \
  --connections 3 \
  --model nova-2 \
  --language en \
  --diarize true \
  --punctuate true \
  --keywords "hello:5,world:10" \
  --loop \
  --duration 60 \
  --region us-east-1 \
  --log-level DEBUG
```

**Available options:**

- `--file WAV_FILE` - Path to the WAV file to stream (required, must be 16-bit PCM)
- `--connections N` - Number of simultaneous streaming connections (default: 1)
- `--model MODEL` - Deepgram model to use (default: nova-3)
- `--language LANG` - Language code (default: en)
- `--diarize true|false` - Enable speaker diarization (default: false)
- `--punctuate true|false` - Enable punctuation (default: true)
- `--keywords KEYWORDS` - Comma-delimited keywords with intensity (format: "word:intensity,word:intensity")
- `--loop` - Loop the WAV file continuously until `--duration` is reached or Ctrl+C
- `--duration SECONDS` - Stop automatically after this many seconds (default: play file once, or loop until Ctrl+C)
- `--region REGION` - AWS region (default: us-east-1)
- `--log-level LEVEL` - Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)


## Test Deepgram Text-to-Speech (TTS) on SageMaker

### JavaScript

TBD

### Python

TBD

## Transcription Load Test

To run transcription load test:

1. Set your AWS credentials (eg. `AWS_SHARED_CREDENTIALS_FILE` and `AWS_PROFILE` variables)
1. Ensure Node.js is installed
1. Set the AWS region, SageMaker Endpoint name, input audio file name, and query string parameters, in `stt.file.ts`
1. Run `npx tsx stress-stt.ts`

