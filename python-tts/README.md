# Deepgram SageMaker Text-to-Speech Clients

Python clients for testing Deepgram Text-to-Speech (TTS) endpoints deployed on AWS SageMaker.

- `tts_http.py` makes a single synchronous HTTP request using the SageMaker `InvokeEndpoint` API.
- `tts_stress.py` streams text phrases over bidirectional HTTP/2 connections for load testing, with audio playback from a single selectable connection.
- `tts_sdk.py` streams one text prompt through the official Deepgram Python SDK using the `deepgram-sagemaker` transport.

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- AWS credentials configured (CLI, environment variables, or IAM role)
- A deployed Amazon SageMaker endpoint running a Deepgram TTS model
- PyAudio for streaming audio playback with `tts_stress.py`:
  - macOS: `brew install portaudio`
  - Linux: `sudo apt-get install portaudio19-dev`

## Installation

```bash
cd python-tts
uv sync
```

---

## `tts_sdk.py`

Synthesizes one text prompt through the official Deepgram Python SDK `speak.v1`
streaming interface and writes the returned audio bytes to disk. This is a small
functional smoke test; use `tts_stress.py` for multi-connection load testing.

This script follows the Deepgram TTS WebSocket contract: it opens the
`/v1/speak` streaming connection, sends a `Speak` message, sends a `Close`
message to flush buffered text and close gracefully after all audio is generated,
and writes binary audio chunks returned by the server. Streaming TTS supports
`linear16`, `mulaw`, and `alaw` output encodings.

If the SageMaker transport surfaces any JSON control payloads as bytes, the
script logs them as events and does not include them in the audio output file.

For the default `linear16` encoding, the output file contains raw PCM bytes. To
play it with FFmpeg:

```bash
ffplay -f s16le -ar 24000 -ac 1 tts-sdk-output.pcm
```

### Examples

**Basic usage:**

```bash
uv run tts_sdk.py your-endpoint-name "Hello from Deepgram TTS on SageMaker."
```

**Read text from a file and choose a voice:**

```bash
uv run tts_sdk.py your-endpoint-name \
  --text-file prompt.txt \
  --model aura-2-thalia-en \
  --output output.pcm
```

**Request mu-law output:**

```bash
uv run tts_sdk.py your-endpoint-name \
  "Hello from an SDK streaming TTS smoke test." \
  --encoding mulaw \
  --sample-rate 8000 \
  --output output.mulaw
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `endpoint_name` | SageMaker endpoint name (required) | — |
| `text` | Text to synthesize when `--text-file` is not used | built-in test sentence |
| `--text-file PATH` | UTF-8 text file to synthesize | — |
| `--output PATH` | Path to write returned audio bytes | `tts-sdk-output.pcm` |
| `--region REGION` | AWS region | `us-east-2` |
| `--model MODEL` | Deepgram TTS voice model | `aura-2-thalia-en` |
| `--encoding ENCODING` | Streaming output audio encoding: `linear16`, `mulaw`, or `alaw` | `linear16` |
| `--sample-rate HZ` | Output sample rate | `24000` |
| `--close-timeout SECONDS` | Wait for the stream to close after `Close` | `30` |

If no audio bytes are written, confirm the endpoint is a TTS endpoint, the
requested voice and encoding are supported by the deployed model, the endpoint is
`InService`, and your IAM principal has
`sagemaker:InvokeEndpointWithBidirectionalStream`.

---

## `tts_stress.py`

Streams text phrases from a file to multiple simultaneous bidirectional connections to a Deepgram TTS endpoint on SageMaker. Audio from one selected connection is played back through local speakers; all other connections receive and discard audio.

### Prepare a text input file

Create a plain text file with one phrase per line (default: `tts-input.txt`):

```
Hello, this is a test of the Deepgram text-to-speech system.
Welcome to the future of voice synthesis.
```

Phrases are cycled repeatedly for the duration of the test.

### Examples

**Basic usage (single connection, 30-second test):**

```bash
uv run tts_stress.py your-endpoint-name
```

**With a specific AWS region:**

```bash
uv run tts_stress.py your-endpoint-name --region us-west-2
```

**Multiple simultaneous connections (load testing):**

```bash
uv run tts_stress.py your-endpoint-name --connections 5
```

**Select which connection plays audio to speakers:**

```bash
uv run tts_stress.py your-endpoint-name --connections 5 --playback 3
```

**With a different Deepgram TTS voice:**

```bash
uv run tts_stress.py your-endpoint-name --voice aura-2-orion-en
```

**Custom duration and text file:**

```bash
uv run tts_stress.py your-endpoint-name --duration 60 --text-file my-phrases.txt
```

**Full example with all options:**

```bash
uv run tts_stress.py your-endpoint-name \
  --connections 5 \
  --playback 2 \
  --duration 120 \
  --voice aura-2-thalia-en \
  --text-file tts-input.txt \
  --region us-east-2 \
  --log-level DEBUG
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `endpoint_name` | SageMaker endpoint name (required) | — |
| `--connections N` | Number of simultaneous streaming connections | `1` |
| `--playback N` | Connection ID whose audio is played to speakers (1-based, must be ≤ connections) | `1` |
| `--duration SECONDS` | How long to run the test | `30` |
| `--voice VOICE` | Deepgram TTS voice model | `aura-2-thalia-en` |
| `--text-file PATH` | Path to a text file with phrases to synthesize, one per line | `tts-input.txt` |
| `--region REGION` | AWS region | `us-east-2` |
| `--log-level LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `INFO` |

### How it works

1. Text phrases are read from the input file and cycled repeatedly for the duration of the test.
2. Each phrase is sent as a `Speak` message followed by a `Flush` to trigger audio synthesis.
3. The script waits for a `Flushed` acknowledgement from every active connection before sending the next phrase, keeping all connections in sync and avoiding server-side rate limiting.
4. The duration timer starts when the first audio chunk is received from the playback connection.
5. On shutdown, the script waits up to 30 seconds for each connection to finish receiving remaining synthesized audio.

---

## `tts_http.py`

Synthesizes text with Deepgram TTS using the SageMaker `InvokeEndpoint` API. This is the HTTP-only, single request/response path and does not use bidirectional streaming or WebSockets.

The script sends a JSON request body in the Deepgram REST format by default:

```json
{"text": "Hello, welcome to Deepgram!"}
```

Deepgram routing and query parameters are passed through SageMaker `CustomAttributes` as:

```text
v1/speak?model=aura-2-thalia-en&encoding=mp3
```

### Examples

**Basic usage:**

```bash
uv run tts_http.py your-endpoint-name "Hello from Deepgram TTS on SageMaker."
```

**Write WAV output with linear16 audio:**

```bash
uv run tts_http.py your-endpoint-name \
  "Hello from a synchronous HTTP InvokeEndpoint request." \
  --encoding linear16 \
  --sample-rate 24000 \
  --output output.wav
```

**Read text from a file:**

```bash
uv run tts_http.py your-endpoint-name --text-file prompt.txt --output speech.mp3
```

**Send a `text/plain` body instead of JSON:**

```bash
uv run tts_http.py your-endpoint-name \
  "Hello from a plain text HTTP InvokeEndpoint request." \
  --body-format text
```

**With a different voice and region:**

```bash
uv run tts_http.py your-endpoint-name \
  "Welcome to the future of voice synthesis." \
  --model aura-2-orion-en \
  --region us-west-2
```

**Full example with all common options:**

```bash
uv run tts_http.py your-endpoint-name \
  --text-file prompt.txt \
  --output speech.mp3 \
  --model aura-2-thalia-en \
  --encoding mp3 \
  --region us-east-2 \
  --log-level DEBUG
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `endpoint_name` | SageMaker endpoint name (required) | - |
| `text` | Text to synthesize when `--text-file` is not used | built-in test sentence |
| `--text-file PATH` | UTF-8 text file to synthesize | - |
| `--output PATH` | Path to write synthesized audio | `tts-output` with an extension based on `--encoding` |
| `--model MODEL` | Deepgram TTS voice model | `aura-2-thalia-en` |
| `--encoding ENCODING` | Output audio encoding: `aac`, `alaw`, `flac`, `linear16`, `mp3`, `mulaw`, `opus` | `mp3` |
| `--sample-rate HZ` | Output sample rate when supported by the encoding; omitted for `mp3` | `24000` |
| `--body-format FORMAT` | Request body format: `json` or `text` | `json` |
| `--bit-rate BPS` | Optional output bit rate for supported encodings | - |
| `--container VALUE` | Optional Deepgram container parameter for supported formats | - |
| `--accept MIME` | Override the SageMaker `Accept` header | inferred from `--encoding` |
| `--region REGION` | AWS region | `us-east-2` |
| `--skip-endpoint-check` | Skip the `DescribeEndpoint` preflight check | disabled |
| `--log-level LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `INFO` |

### How it works

1. Text is read from the positional argument or `--text-file`.
2. The script builds the Deepgram REST path and query string as `v1/speak?...`.
3. It calls SageMaker Runtime `InvokeEndpoint` with a JSON or text `ContentType`, an audio `Accept` header, and the Deepgram path/query string in `CustomAttributes`.
4. The response body is written directly to the requested output audio file.

---

## Troubleshooting

### Audio not playing

- Verify `--playback` is between 1 and the number of `--connections`
- Check system audio output settings
- Verify PyAudio is installed: `python -c "import pyaudio; print('OK')"`
- macOS: `brew install portaudio && pip install pyaudio`

### Connection errors

- Verify the endpoint name is correct and matches the target `--region`
- Confirm AWS credentials are configured: `aws sts get-caller-identity`
- Confirm the endpoint is `InService` in the AWS Console or via `aws sagemaker describe-endpoint --endpoint-name your-endpoint-name`
- Check CloudWatch Logs for the SageMaker endpoint for server-side errors
- For `tts_http.py`, confirm the IAM principal has `sagemaker:InvokeEndpoint`
- For `tts_stress.py`, confirm the IAM principal has `sagemaker:InvokeEndpointWithBidirectionalStream`

### No audio output

- Ensure the `--voice` is valid for the deployed TTS model — see the [Voices documentation](https://developers.deepgram.com/docs/tts-models)
- Check system volume and speaker settings
- Enable debug logging: `--log-level DEBUG`

### Empty or invalid HTTP output

- For `tts_http.py`, verify the endpoint is a TTS endpoint and supports the REST `/v1/speak` path
- Confirm the requested `--encoding`, `--sample-rate`, and optional `--bit-rate` combination is supported
- Try `--log-level DEBUG` and inspect the SageMaker endpoint's CloudWatch logs for model-side errors
