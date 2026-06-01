# Deepgram SageMaker Speech-to-Text Stress Test Client

Python scripts for stress testing Deepgram Speech-to-Text (STT) endpoints deployed on AWS SageMaker. Three input modes are supported:

- **`stt_microphone_stress.py`** — streams live microphone audio for real-time transcription
- **`stt_wav_stress.py`** — streams a WAV file or sends synchronous batch HTTP requests; supports multiple simultaneous connections for load testing
- **`stt_wav_async.py`** — transcribes a WAV file via SageMaker `InvokeEndpointAsync` (S3 in/out); the right path for files larger than the 25 MB synchronous limit (up to 1 GiB), for jobs that need up to 60 min of processing time, or need the endpoint to scale to zero when there are no requests being processed.

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

**Exercise the bare WebSocket Close path (skip CloseStream):**

By default each session sends a Deepgram [`CloseStream`](https://developers.deepgram.com/docs/close-stream) message before closing the WebSocket so the server flushes the final transcript tail. Pass `--no-use-close-stream` to instead close with a bare WS Close frame (the trailing transcript is dropped when `endpointing=false`):

```bash
uv run stt_wav_stress.py stream your-endpoint-name --file audio.wav \
  --no-use-close-stream
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
| `--use-close-stream` / `--no-use-close-stream` | Send a Deepgram [`CloseStream`](https://developers.deepgram.com/docs/close-stream) message before the WebSocket Close so the server flushes the final transcript tail; `--no-use-close-stream` exercises the bare WS Close path instead | on |
| `--loop` | Loop the WAV file until `--duration` is reached or Ctrl+C | off |
| `--duration SECONDS` | Stop after this many seconds | play file once |
| `--region REGION` | AWS region | `us-east-2` |
| `--log-level LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `INFO` |

---

### `batch` — Pre-recorded HTTP transcription

Posts the entire WAV file in a single HTTP request using the SageMaker `InvokeEndpoint` API. Supports configurable parallelism via `--concurrency` for throughput and latency stress testing. Each concurrent request runs on its own Python thread with its own boto3 client. After all requests complete, a summary table shows min/avg/p95/max latency, throughput, and success/failure counts.

> **Note:** SageMaker `InvokeEndpoint` has a 25 MB request body limit. For larger files, use `stream` mode or split the file:
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
| `--file WAV_FILE` | Path to a 16-bit PCM WAV file, max 25 MB (required) | — |
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

---

## `stt_wav_async.py`

Transcribes a WAV file via the SageMaker `InvokeEndpointAsync` API (S3 input → S3 output). Use this when:

- The audio is larger than the **25 MB** synchronous `InvokeEndpoint` body limit, **and/or**
- Inference will take longer than the synchronous 60-second wall-clock budget.

Async inference accepts S3 objects up to **1 GiB** and gives each invocation up to **60 minutes** of processing time. The script uploads the WAV to S3 (or accepts an existing S3 URI), calls `InvokeEndpointAsync`, and polls the configured output + failure prefixes for the result. Multiple invocations can run in parallel via `--concurrency`.

### Endpoint prerequisites

The target endpoint must be created from an `EndpointConfig` that carries an `AsyncInferenceConfig`, for example:

```json
"AsyncInferenceConfig": {
  "OutputConfig": {
    "S3OutputPath":  "s3://your-async-bucket-name/output/",
    "S3FailurePath": "s3://your-async-bucket-name/output/failures/"
  },
  "ClientConfig": { "MaxConcurrentInvocationsPerInstance": 4 }
}
```

The SageMaker execution role attached to the endpoint also needs:

- `s3:GetObject` on the input prefix used by this script
- `s3:PutObject` on the output + failures prefixes
- `s3:ListBucket` on the bucket

Note: `AsyncInferenceConfig` is incompatible with `EnableSSMAccess=true` on the same production variant — pick one or the other.

The IAM identity that **runs the script** only needs:

- `s3:PutObject` on the chosen upload key (skipped when `--input-s3-uri` is supplied)
- `s3:GetObject` on the resulting success / failure objects
- `sagemaker:InvokeEndpointAsync` on the endpoint

### Examples

**Basic usage (upload + transcribe a long file):**

```bash
uv run stt_wav_async.py your-async-endpoint-name \
  --file long-recording.wav \
  --bucket your-async-bucket-name
```

**Reuse an existing S3 object (skip upload):**

```bash
uv run stt_wav_async.py your-async-endpoint-name \
  --file long-recording.wav \
  --input-s3-uri s3://your-async-bucket-name/input/long-recording.wav
```

**Different model and language:**

```bash
uv run stt_wav_async.py your-async-endpoint-name \
  --file audio.wav --bucket your-async-bucket-name \
  --model nova-2 --language es
```

**Diarization + keyterms:**

```bash
uv run stt_wav_async.py your-async-endpoint-name \
  --file audio.wav --bucket your-async-bucket-name \
  --diarize true --keyterms "Deepgram,SageMaker"
```

**Throughput test — 20 parallel invocations of the same file:**

```bash
uv run stt_wav_async.py your-async-endpoint-name \
  --file audio.wav --bucket your-async-bucket-name \
  --concurrency 20 --requests 20
```

**Long-running invocation with a custom per-request timeout:**

```bash
uv run stt_wav_async.py your-async-endpoint-name \
  --file audio.wav --bucket your-async-bucket-name \
  --invocation-timeout 3600 --poll-interval 10
```

**Extra Deepgram parameters (sentiment, topics, etc.):**

```bash
uv run stt_wav_async.py your-async-endpoint-name \
  --file audio.wav --bucket your-async-bucket-name \
  --extra "sentiment=true&topics=true"
```

**Full example with all options:**

```bash
uv run stt_wav_async.py your-async-endpoint-name \
  --file long-recording.wav \
  --bucket your-async-bucket-name \
  --upload-prefix stt-async-input \
  --model nova-3 \
  --language en \
  --diarize true \
  --punctuate true \
  --keyterms "Deepgram,SageMaker" \
  --redact "pii,ssn" \
  --extra "sentiment=true&topics=true" \
  --concurrency 4 \
  --requests 8 \
  --invocation-timeout 3600 \
  --poll-interval 5 \
  --inference-id-prefix bug-bash-2026-06 \
  --region us-east-2 \
  --log-level INFO
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `endpoint_name` | SageMaker endpoint name (required) | — |
| `--file WAV_FILE` | Path to a 16-bit PCM WAV file (up to 1 GiB) | — |
| `--bucket BUCKET` | Upload-to bucket — object lands at `s3://<bucket>/<upload-prefix>/<uuid>.wav`. **Mutually exclusive with `--input-s3-uri`** | — |
| `--input-s3-uri S3_URI` | Use an existing S3 object instead of uploading. **Mutually exclusive with `--bucket`** | — |
| `--upload-prefix PREFIX` | Key prefix used with `--bucket` | `stt-async-input` |
| `--model MODEL` | Deepgram model | `nova-3` |
| `--language LANG` | Language code | `en` |
| `--diarize true\|false` | Enable speaker diarization | `false` |
| `--punctuate true\|false` | Enable punctuation | `true` |
| `--keyterms TERM,...` | Comma-separated keyterms to boost recognition (nova-3) | — |
| `--redact ENTITY,...` | Comma-separated entity types to redact, e.g. `pii,ssn,email_address` | — |
| `--extra k=v&k2=v2` | Extra Deepgram query parameters appended verbatim | — |
| `--concurrency N` | Invocations to keep in flight in parallel | `1` |
| `--requests N` | Total number of invocations to submit | same as `--concurrency` |
| `--invocation-timeout SECONDS` | Per-invocation timeout passed to SageMaker (max 3600) | `3600` |
| `--poll-interval SECONDS` | Seconds between S3 polls for each invocation's result | `5.0` |
| `--inference-id-prefix PREFIX` | Override the `InferenceId` prefix sent to SageMaker | random per run |
| `--show-transcript` / `--no-show-transcript` | Print transcript head per successful invocation | on |
| `--transcript-chars N` | Characters of each transcript to print | `300` |
| `--region REGION` | AWS region | `us-east-2` |
| `--log-level LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `INFO` |

### Output

Each successful invocation is summarized as:

```
[Request   1] ✓ (24.83s) duration=1791.53s conf=99.5% output=s3://your-async-bucket-name/output/3051404d-7473-4e61-ba37-74f31f1bf3ed.out
             november the tenth wednesday nine pm i'm standing in a dark alley after waiting several hours…
```

`duration` is the audio duration the container reported; `elapsed` is the wall-clock submit-to-result time observed by this script (includes queue wait + inference + polling slack). The full JSON response stays in S3 at the listed `output=` URI for follow-up inspection (e.g. `aws s3 cp <uri> - | jq`).

Failures are written to the `S3FailurePath` configured on the EndpointConfig and printed as:

```
[Request   2] ERROR (1.34s): Endpoint reported failure: billing: nats grace window exceeded
             output uri: s3://your-async-bucket-name/output/failures/8ef24943-a086-45f8-87c6-2cba6f37aaa7-error.out
```
