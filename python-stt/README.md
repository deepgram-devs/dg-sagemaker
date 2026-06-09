# Deepgram SageMaker Speech-to-Text Stress Test Client

Python scripts for stress testing Deepgram Speech-to-Text (STT) endpoints deployed on AWS SageMaker. Three input modes are supported:

- **`stt_microphone_stress.py`** — streams live microphone audio for real-time transcription
- **`stt_wav_stress.py`** — streams a WAV file or sends synchronous batch HTTP requests; supports multiple simultaneous connections for load testing
- **`stt_wav_async.py`** — transcribes a WAV file via SageMaker `InvokeEndpointAsync` (S3 in/out); the right path for files larger than the 25 MB synchronous limit (up to 1 GiB), for jobs that need up to 60 min of processing time, or need the endpoint to scale to zero when there are no requests being processed.

Two **end-to-end correctness drivers** sit on top of these scripts (in `e2e/`) as definitive promotion gates:

- **`e2e/e2e_test_streaming.py`** — downloads `https://dpgr.am/spacewalk.wav`, multiplies it to ~15 min, drives `stt_wav_stress.py stream` through 10 scenarios (basic / sustained-concurrency / ramped / feature flags / adversarial close path), and validates every connection's `combined_final_text` against a known reference transcript via Word Error Rate.
- **`e2e/e2e_test_batch.py`** — same fixtures, takes `--mode sync` or `--mode async` to match how the target endpoint is configured (a single batch endpoint serves only one transport). Sync mode runs five scenarios on the 25-second sample (≤ 25 MB body limit); async mode runs five scenarios on the long-form 15-min variant via S3 in/out (~76 MB, well over the sync cap), including summarize via fathom.

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

---

## End-to-end correctness drivers (`e2e/`)

Run-everything scripts that exercise an endpoint across many scenarios and gate promotion on transcript correctness. Both download `https://dpgr.am/spacewalk.wav` (~25 s English mono) once, multiply it to a ~15 min long-form variant on first run, then validate each session's transcript against the canonical reference text via Word Error Rate (WER ≤ 5 % by default; relaxed per-scenario where transcription is intentionally distorted, e.g. redact).

The drivers live in `e2e/` alongside `e2e_test_common.py` (shared helpers); the parent directory holds the core stress scripts they wrap.

### `e2e/e2e_test_streaming.py`

Definitive correctness gate for a **streaming** endpoint. Subprocesses `stt_wav_stress.py stream` ten times with different argument sets — short-form, long-form, sustained-concurrency, ramped-concurrency, each major feature flag, and an adversarial bare-WS-close path — and reads each connection's `combined_final_text` out of the per-conn `--summary-jsonl` to compute WER.

```bash
uv run e2e/e2e_test_streaming.py your-streaming-endpoint-name --region us-east-2

# list scenarios without running:
uv run e2e/e2e_test_streaming.py --list

# run a single scenario:
uv run e2e/e2e_test_streaming.py your-endpoint --scenarios basic_25s

# tighten the global threshold to 3 %:
uv run e2e/e2e_test_streaming.py your-endpoint --wer-threshold 0.03
```

| Scenario | What it checks |
|---|---|
| `basic_25s` | One connection, 25 s file, defaults — baseline WER |
| `basic_15min` | One connection, 15 min looped file — long-form smoke |
| `concurrent_5x_25s` | 5 simultaneous connections, 25 s file |
| `concurrent_10x_15min` | 10 simultaneous connections, 15 min file — sustained load |
| `ramp_10x_step5` | 10 conns in batches of 5 with 2 s delay |
| `feature_interim` | `--interim-results true` — verifies interims emitted |
| `feature_diarize` | `--diarize true` — body unchanged, speaker tags added |
| `feature_keyterms` | `--keyterms 'spacewalk,female'` — presence check on `spacewalk` |
| `feature_redact_name` | `--redact name` — presence of redaction marker, WER skipped |
| `adversarial_bare_close` | `--no-use-close-stream` — bare WS close (relaxed WER 10 %) |
| `reject_unknown_param` | `--extra bogus=true` — negative: PASSES only if the upgrade is rejected (400) and no audio streams; verifies the shim's reject-unknown-params gate |

Exit code 0 = all pass, non-zero = any scenario failed. Per-scenario stdout / stderr / summary-jsonl land in `<workdir>/logs/<scenario>.{stdout,stderr,summary.jsonl}.log` for triage; aggregated `results.json` at `<workdir>/results.json`. Default workdir: `/tmp/dg-sagemaker-e2e/streaming/<timestamp>/`.

### `e2e/e2e_test_batch.py`

Definitive correctness gate for a **batch** endpoint. A batch endpoint serves one of two transports — sync `invoke_endpoint` or async `invoke_endpoint_async` — based on whether its EndpointConfig includes an `AsyncInferenceConfig` block. `--mode {sync,async}` picks the matching scenario set; the driver refuses to run a sync scenario set against an async-configured endpoint or vice versa.

```bash
# sync-configured endpoint (no AsyncInferenceConfig):
uv run e2e/e2e_test_batch.py your-sync-endpoint --mode sync --region us-east-2

# async-configured endpoint (with AsyncInferenceConfig):
uv run e2e/e2e_test_batch.py your-async-endpoint --mode async \
  --bucket your-async-bucket --region us-east-2

# subset (must still match the mode):
uv run e2e/e2e_test_batch.py your-async-endpoint --mode async \
  --bucket your-async-bucket --scenarios async_25s,async_15min

# list scenarios for one mode:
uv run e2e/e2e_test_batch.py --mode async --list
```

| Scenario | Transport | What it checks |
|---|---|---|
| `sync_25s` | sync | Baseline WER on 25 s file |
| `sync_25s_concurrent_5` | sync | 5 concurrent invokes |
| `sync_25s_diarize` | sync | `diarize=true` — body unchanged |
| `sync_25s_keyterms` | sync | keyterms biasing, presence check |
| `sync_25s_redact_name` | sync | `redact=name`, WER skipped |
| `sync_25s_reject_unknown_param` | sync | negative: `bogus=true` must 400 (`unsupported_parameter`), not serve — reject-unknown-params gate |
| `sync_25s_reject_unknown_param_falsy` | sync | negative: `bogus=false` must also 400 (reject is value-independent) |
| `async_25s` | async | Short-form smoke (25 s file via S3) |
| `async_15min` | async | 15 min audio via S3 in/out |
| `async_15min_concurrent_4` | async | 4 concurrent async invokes |
| `async_15min_diarize` | async | Diarize on long audio |
| `async_15min_summarize` | async | `summarize=v2` — requires fathom in bundle |

`--bucket` is required with `--mode async` (where input WAVs are uploaded before each invocation) and ignored with `--mode sync`. Same exit-code semantics and `results.json` layout as the streaming driver.

### Common options

Both drivers accept:

| Option | Default | Description |
|---|---|---|
| `--region` | `us-east-2` | AWS region |
| `--model` / `--language` | `nova-3` / `en` | passed through to each scenario |
| `--workdir DIR` | `/tmp/dg-sagemaker-e2e/<kind>/<ts>` | fixtures + logs |
| `--target-long-form-s` | `900` (15 min) | duration of the multiplied long-form WAV |
| `--scenarios` | (all) | comma-separated subset; use `--list` to discover names |
| `--wer-threshold` | `0.05` | default per-scenario threshold (some scenarios override) |
| `--force-download` | off | re-download spacewalk.wav even if cached |
