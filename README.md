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

The `stt_wav_stress.py` script supports two sub-commands: `stream` and `batch`.

> **Requirements:** The WAV file must be 16-bit PCM (linear16). To convert any audio file, use `ffmpeg`:
> ```bash
> ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 output.wav
> ```

---

##### `stream` — Real-time bidirectional streaming

Streams the WAV file to Deepgram on SageMaker in real-time, paced to match the file's actual sample rate. Behaves like a live microphone source, enabling repeatable and automated load testing without requiring a physical microphone.

**Basic usage (single connection, play file once):**

```bash
uv run python-stt/stt_wav_stress.py stream your-endpoint-name --file audio.wav
```

**With a specific AWS region:**

```bash
uv run python-stt/stt_wav_stress.py stream your-endpoint-name --file audio.wav --region us-west-2
```

**Multiple simultaneous connections (load testing):**

```bash
uv run python-stt/stt_wav_stress.py stream your-endpoint-name --file audio.wav --connections 5
```

**Loop the file continuously until Ctrl+C:**

```bash
uv run python-stt/stt_wav_stress.py stream your-endpoint-name --file audio.wav --loop
```

**Loop for a fixed duration (useful for automated tests):**

```bash
uv run python-stt/stt_wav_stress.py stream your-endpoint-name --file audio.wav --loop --duration 120
```

**Timed load test with multiple connections:**

```bash
uv run python-stt/stt_wav_stress.py stream your-endpoint-name --file audio.wav \
  --connections 10 --loop --duration 300
```

**With speaker diarization:**

```bash
uv run python-stt/stt_wav_stress.py stream your-endpoint-name --file audio.wav --diarize true
```

**With a different model and language:**

```bash
uv run python-stt/stt_wav_stress.py stream your-endpoint-name --file audio.wav \
  --model nova-2 --language es
```

**With keywords boosting (nova-2 only) or keyterms (nova-3):**

Keywords are only compatible with `nova-2`. For `nova-3` use `--keyterms` instead.

```bash
# nova-2 keywords
uv run python-stt/stt_wav_stress.py stream your-endpoint-name --file audio.wav \
  --keywords "Deepgram:5,SageMaker:10"

# nova-3 keyterms
uv run python-stt/stt_wav_stress.py stream your-endpoint-name --file audio.wav \
  --keyterms "Deepgram,SageMaker"
```

**With PII redaction:**

```bash
uv run python-stt/stt_wav_stress.py stream your-endpoint-name --file audio.wav \
  --redact "pii,ssn,email_address"
```

**Full example with all stream options:**

```bash
uv run python-stt/stt_wav_stress.py stream your-endpoint-name --file audio.wav \
  --connections 3 \
  --model nova-3 \
  --language en \
  --diarize true \
  --punctuate true \
  --keyterms "Deepgram,SageMaker" \
  --redact "pii,ssn" \
  --interim-results true \
  --loop \
  --duration 60 \
  --region us-east-2 \
  --log-level DEBUG
```

**Available `stream` options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--file WAV_FILE` | Path to a 16-bit PCM WAV file (required) | — |
| `--connections N` | Number of simultaneous streaming connections | `1` |
| `--model MODEL` | Deepgram model | `nova-3` |
| `--language LANG` | Language code | `en` |
| `--diarize true\|false` | Enable speaker diarization | `false` |
| `--punctuate true\|false` | Enable punctuation | `true` |
| `--keywords WORD:N,...` | Keyword boosting with intensity, e.g. `hello:5,world:10` (nova-2 only) | — |
| `--keyterms TERM,...` | Comma-separated keyterms to boost recognition (nova-3) | — |
| `--redact ENTITY,...` | Comma-separated entity types to redact, e.g. `pii,ssn,email_address` | — |
| `--interim-results true\|false` | Emit interim (partial) transcripts | `true` |
| `--loop` | Loop the WAV file until `--duration` is reached or Ctrl+C | off |
| `--duration SECONDS` | Stop after this many seconds | play file once |
| `--region REGION` | AWS region | `us-east-2` |
| `--log-level LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `INFO` |

---

##### `batch` — Pre-recorded HTTP transcription

Posts the entire WAV file in a single HTTP request using the SageMaker `InvokeEndpoint` API. Supports configurable parallelism via `--concurrency` for throughput and latency stress testing. Each concurrent request runs on its own Python thread with its own boto3 client. After all requests complete, a summary table shows min/avg/p95/max latency, throughput, and success/failure counts.

> **Note:** SageMaker `InvokeEndpoint` has a 6 MB request body limit. For larger files, use `stream` mode or split the file:
> ```bash
> ffmpeg -i input.wav -f segment -segment_time 60 segment_%03d.wav
> ```

**Basic usage (single request):**

```bash
uv run python-stt/stt_wav_stress.py batch your-endpoint-name --file audio.wav
```

**Send 10 concurrent requests (load testing):**

```bash
uv run python-stt/stt_wav_stress.py batch your-endpoint-name --file audio.wav --concurrency 10
```

**Send 100 total requests, 10 at a time:**

```bash
uv run python-stt/stt_wav_stress.py batch your-endpoint-name --file audio.wav \
  --concurrency 10 --requests 100
```

**With a different model and language:**

```bash
uv run python-stt/stt_wav_stress.py batch your-endpoint-name --file audio.wav \
  --model nova-2 --language es
```

**With keyterms (nova-3):**

```bash
uv run python-stt/stt_wav_stress.py batch your-endpoint-name --file audio.wav \
  --keyterms "Deepgram,SageMaker"
```

**With speaker diarization and PII redaction:**

```bash
uv run python-stt/stt_wav_stress.py batch your-endpoint-name --file audio.wav \
  --diarize true --redact "pii,ssn,email_address"
```

**Full example with all batch options:**

```bash
uv run python-stt/stt_wav_stress.py batch your-endpoint-name --file audio.wav \
  --concurrency 5 \
  --requests 50 \
  --model nova-3 \
  --language en \
  --diarize true \
  --punctuate true \
  --keyterms "Deepgram,SageMaker" \
  --redact "pii,ssn" \
  --region us-east-2 \
  --log-level DEBUG
```

**Available `batch` options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--file WAV_FILE` | Path to a 16-bit PCM WAV file, max 6 MB (required) | — |
| `--concurrency N` | Number of requests to run in parallel | `1` |
| `--requests N` | Total number of requests to send | same as `--concurrency` |
| `--model MODEL` | Deepgram model | `nova-3` |
| `--language LANG` | Language code | `en` |
| `--diarize true\|false` | Enable speaker diarization | `false` |
| `--punctuate true\|false` | Enable punctuation | `true` |
| `--keyterms TERM,...` | Comma-separated keyterms to boost recognition (nova-3) | — |
| `--redact ENTITY,...` | Comma-separated entity types to redact, e.g. `pii,ssn,email_address` | — |
| `--region REGION` | AWS region | `us-east-2` |
| `--log-level LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `INFO` |


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

