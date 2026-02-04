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

**Full example with all options:**

```bash
uv run python-stt/stt_microphone_stress.py your-endpoint-name \
  --connections 3 \
  --model nova-2 \
  --language en \
  --diarize true \
  --punctuate true \
  --keywords "hello:5,world:10" \
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

