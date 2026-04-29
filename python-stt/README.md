# Deepgram SageMaker Speech-to-Text Stress Test Client

Python scripts for stress testing Deepgram Speech-to-Text (STT) endpoints deployed on AWS SageMaker. Two input modes are supported:

- **`stt_microphone_stress.py`** — streams live microphone audio for real-time transcription
- **`stt_wav_stress.py`** — streams a WAV file or sends batch HTTP requests; supports multiple simultaneous connections for load testing
- **`stt_sdk.py`** — streams a WAV file through the official Deepgram Python SDK using the `deepgram-sagemaker` transport

## Prerequisites

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) package manager
- AWS credentials configured (CLI, environment variables, or IAM role)
- A deployed Amazon SageMaker endpoint running a Deepgram STT model

## Installation

```bash
cd python-stt
uv sync
```

**macOS — microphone support requires PortAudio:**

```bash
brew install portaudio
uv sync
```

---

## `stt_sdk.py`

Streams one 16-bit PCM WAV file to a Nova STT SageMaker endpoint through the
official Deepgram Python SDK. This is intended as a small functional smoke test;
use `stt_wav_stress.py` or `stt_microphone_stress.py` for load testing.

### Examples

**Basic usage:**

```bash
uv run stt_sdk.py your-endpoint-name --file audio.wav
```

**With nova-3 keyterms:**

```bash
uv run stt_sdk.py your-endpoint-name \
  --file audio.wav \
  --keyterms "Deepgram,SageMaker" \
  --region us-east-2
```

**With diarization and final-only style output from Deepgram disabled:**

```bash
uv run stt_sdk.py your-endpoint-name \
  --file audio.wav \
  --diarize true \
  --interim-results true
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `endpoint_name` | SageMaker endpoint name (required) | — |
| `--file WAV_FILE` | 16-bit PCM WAV file to stream (required) | — |
| `--region REGION` | AWS region | `us-east-2` |
| `--model MODEL` | Deepgram STT model | `nova-3` |
| `--language LANG` | Language code | `en` |
| `--punctuate true\|false` | Enable punctuation | `true` |
| `--interim-results true\|false` | Emit interim transcripts | `true` |
| `--diarize true\|false` | Enable speaker diarization | `false` |
| `--keywords WORD:N,...` | Keyword boosting for nova-2 | — |
| `--keyterms TERM,...` | Keyterm prompting for nova-3 | — |
| `--chunk-ms N` | Audio chunk duration | `80` |
| `--drain-seconds N` | Wait for trailing responses before exit | `2` |

The script uses AWS credentials from the standard credential chain. If it fails
before streaming starts, confirm the endpoint exists in the selected region, is
`InService`, and your IAM principal has
`sagemaker:InvokeEndpointWithBidirectionalStream`.

---

## `stt_microphone_stress.py`

Streams live microphone audio to Deepgram on SageMaker for real-time transcription. Supports multiple simultaneous connections for load testing.

### Examples

**Basic usage (single connection):**

```bash
uv run stt_microphone_stress.py your-endpoint-name
```

**With a specific AWS region:**

```bash
uv run stt_microphone_stress.py your-endpoint-name --region us-west-2
```

**Multiple simultaneous connections (load testing):**

```bash
uv run stt_microphone_stress.py your-endpoint-name --connections 5
```

**With speaker diarization:**

```bash
uv run stt_microphone_stress.py your-endpoint-name --diarize true
```

**With a different model and language:**

```bash
uv run stt_microphone_stress.py your-endpoint-name --model nova-2 --language es
```

**With keywords boosting (nova-2 only):**

```bash
uv run stt_microphone_stress.py your-endpoint-name --keywords "Deepgram:5,SageMaker:10,transcription:3"
```

**Run for a fixed duration (useful for automated tests):**

```bash
uv run stt_microphone_stress.py your-endpoint-name --duration 30
```

**Timed load test with multiple connections:**

```bash
uv run stt_microphone_stress.py your-endpoint-name --connections 5 --duration 120
```

**Full example with all options:**

```bash
uv run stt_microphone_stress.py your-endpoint-name \
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

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `endpoint_name` | SageMaker endpoint name (required) | — |
| `--connections N` | Number of simultaneous streaming connections | `1` |
| `--model MODEL` | Deepgram model | `nova-3` |
| `--language LANG` | Language code | `en` |
| `--diarize true\|false` | Enable speaker diarization | `false` |
| `--punctuate true\|false` | Enable punctuation | `true` |
| `--keywords WORD:N,...` | Keyword boosting with intensity, e.g. `hello:5,world:10` (nova-2 only) | — |
| `--duration SECONDS` | Stop after this many seconds | run until Ctrl+C |
| `--region REGION` | AWS region | `us-east-1` |
| `--log-level LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `INFO` |

---

## `stt_wav_stress.py`

Supports two sub-commands: `stream` and `batch`.

> **Requirements:** The WAV file must be 16-bit PCM (linear16). To convert any audio file, use `ffmpeg`:
> ```bash
> ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 output.wav
> ```

---

### `stream` — Real-time bidirectional streaming

Streams the WAV file to Deepgram on SageMaker in real-time, paced to match the file's actual sample rate. Behaves like a live microphone source, enabling repeatable and automated load testing without requiring a physical microphone.

#### Examples

**Basic usage (single connection, play file once):**

```bash
uv run stt_wav_stress.py stream your-endpoint-name --file audio.wav
```

**With a specific AWS region:**

```bash
uv run stt_wav_stress.py stream your-endpoint-name --file audio.wav --region us-west-2
```

**Multiple simultaneous connections (load testing):**

```bash
uv run stt_wav_stress.py stream your-endpoint-name --file audio.wav --connections 5
```

**Loop the file continuously until Ctrl+C:**

```bash
uv run stt_wav_stress.py stream your-endpoint-name --file audio.wav --loop
```

**Loop for a fixed duration (useful for automated tests):**

```bash
uv run stt_wav_stress.py stream your-endpoint-name --file audio.wav --loop --duration 120
```

**Timed load test with multiple connections:**

```bash
uv run stt_wav_stress.py stream your-endpoint-name --file audio.wav \
  --connections 10 --loop --duration 300
```

**Gradual ramp-up (open 5 connections at a time, 3 seconds apart):**

```bash
uv run stt_wav_stress.py stream your-endpoint-name --file audio.wav \
  --connections 20 --batch-size 5 --batch-delay 3
```

**With speaker diarization:**

```bash
uv run stt_wav_stress.py stream your-endpoint-name --file audio.wav --diarize true
```

**With a different model and language:**

```bash
uv run stt_wav_stress.py stream your-endpoint-name --file audio.wav \
  --model nova-2 --language es
```

**With keywords boosting (nova-2 only) or keyterms (nova-3):**

```bash
# nova-2 keywords
uv run stt_wav_stress.py stream your-endpoint-name --file audio.wav \
  --keywords "Deepgram:5,SageMaker:10"

# nova-3 keyterms
uv run stt_wav_stress.py stream your-endpoint-name --file audio.wav \
  --keyterms "Deepgram,SageMaker"
```

**With PII redaction:**

```bash
uv run stt_wav_stress.py stream your-endpoint-name --file audio.wav \
  --redact "pii,ssn,email_address"
```

**Full example with all stream options:**

```bash
uv run stt_wav_stress.py stream your-endpoint-name --file audio.wav \
  --connections 20 \
  --batch-size 5 \
  --batch-delay 3 \
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

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--file WAV_FILE` | Path to a 16-bit PCM WAV file (required) | — |
| `--connections N` | Total number of simultaneous streaming connections | `1` |
| `--batch-size N` | Connections to open per batch; streaming begins immediately for each batch | all at once |
| `--batch-delay SECONDS` | Seconds to wait between opening connection batches | `0` |
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

### `batch` — Pre-recorded HTTP transcription

Posts the entire WAV file in a single HTTP request using the SageMaker `InvokeEndpoint` API. Supports configurable parallelism via `--concurrency` for throughput and latency stress testing. Each concurrent request runs on its own Python thread with its own boto3 client. After all requests complete, a summary table shows min/avg/p95/max latency, throughput, and success/failure counts.

> **Note:** SageMaker `InvokeEndpoint` has a 6 MB request body limit. For larger files, use `stream` mode or split the file:
> ```bash
> ffmpeg -i input.wav -f segment -segment_time 60 segment_%03d.wav
> ```

#### Examples

**Basic usage (single request):**

```bash
uv run stt_wav_stress.py batch your-endpoint-name --file audio.wav
```

**Send 10 concurrent requests (load testing):**

```bash
uv run stt_wav_stress.py batch your-endpoint-name --file audio.wav --concurrency 10
```

**Send 100 total requests, 10 at a time:**

```bash
uv run stt_wav_stress.py batch your-endpoint-name --file audio.wav \
  --concurrency 10 --requests 100
```

**With a different model and language:**

```bash
uv run stt_wav_stress.py batch your-endpoint-name --file audio.wav \
  --model nova-2 --language es
```

**With keyterms (nova-3):**

```bash
uv run stt_wav_stress.py batch your-endpoint-name --file audio.wav \
  --keyterms "Deepgram,SageMaker"
```

**With speaker diarization and PII redaction:**

```bash
uv run stt_wav_stress.py batch your-endpoint-name --file audio.wav \
  --diarize true --redact "pii,ssn,email_address"
```

**Full example with all batch options:**

```bash
uv run stt_wav_stress.py batch your-endpoint-name --file audio.wav \
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

#### Options

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
