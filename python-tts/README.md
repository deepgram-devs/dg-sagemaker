# Deepgram SageMaker Text-to-Speech Stress Test Client

A Python client for stress testing Deepgram Text-to-Speech (TTS) endpoints deployed on AWS SageMaker. Streams text phrases to multiple simultaneous bidirectional connections for load testing, with audio playback from a single selectable connection.

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- AWS credentials configured (CLI, environment variables, or IAM role)
- A deployed Amazon SageMaker endpoint running a Deepgram TTS model
- PyAudio for audio playback:
  - macOS: `brew install portaudio`
  - Linux: `sudo apt-get install portaudio19-dev`

## Installation

```bash
cd python-tts
uv sync
```

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
| `--playback N` | Connection ID whose audio is played to speakers (1-based, ≤ connections; `0` = headless) | `1` |
| `--no-playback` | Run headless — no speaker playback (sets `--playback 0`). Required when PortAudio/pyaudio is unavailable (CI / e2e) | off |
| `--duration SECONDS` | How long to run the test | `30` |
| `--once` | Send each phrase exactly once then stop (deterministic), instead of cycling for `--duration` | off |
| `--voice VOICE` | Deepgram TTS voice model | `aura-2-thalia-en` |
| `--text-file PATH` | Path to a text file with phrases to synthesize, one per line | `tts-input.txt` |
| `--text "..."` | Inline text to synthesize (overrides `--text-file`; single phrase) | — |
| `--extra "k=v&k2=v2"` | Extra `/v1/speak` query params appended verbatim (e.g. `encoding=mulaw&sample_rate=8000&speed=1.2`) | — |
| `--summary-jsonl PATH` | Write a per-connection JSON summary (audio bytes/RMS/duration, `Flushed` acks, warnings, errors) — consumed by the e2e driver | — |
| `--save-audio-dir DIR` | Save each connection's synthesized audio to DIR (WAV for `linear16`, raw bytes otherwise) | — |
| `--region REGION` | AWS region | `us-east-2` |
| `--log-level LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `INFO` |

### How it works

1. Text phrases are read from the input file and cycled repeatedly for the duration of the test.
2. Each phrase is sent as a `Speak` message followed by a `Flush` to trigger audio synthesis.
3. The script waits for a `Flushed` acknowledgement from every active connection before sending the next phrase, keeping all connections in sync and avoiding server-side rate limiting.
4. The duration timer starts when the first audio chunk is received from the playback connection.
5. On shutdown, the script waits up to 30 seconds for each connection to finish receiving remaining synthesized audio.

---

## End-to-end correctness drivers (`e2e/`)

Run-everything correctness gates that exercise a TTS endpoint across its full
parameter surface and validate the **synthesized audio self-contained** — TTS
has no transcript to score against, so each scenario checks the audio itself
(non-empty bytes, correct container/codec, non-silent RMS for `linear16`, the
requested sample rate, and — for speed control — that duration changes with
`speed`). No second (STT) endpoint is required.

A TTS endpoint serves one of two transports, so there are two drivers (mirroring
the STT split):

### `e2e/e2e_test_batch.py` — REST (`invoke_endpoint`)

The bulk of parameter coverage. Calls `invoke_endpoint` synchronously with the
`/v1/speak` path + query in `CustomAttributes` and a JSON `{"text": "..."}` body,
then validates the returned audio.

```bash
cd python-tts
uv run e2e/e2e_test_batch.py your-tts-endpoint --region us-east-2
uv run e2e/e2e_test_batch.py --list
```

| Scenario | What it checks |
|---|---|
| `basic` / `concurrent_5` | non-empty, non-silent audio; 5-way concurrency |
| `voice_aura2_orion` / `voice_aura1_asteria` | alternate Aura-2 / legacy Aura-1 voices (PASS-WITH-NOTE if unbundled) |
| `encoding_linear16_wav` / `encoding_linear16_raw` | WAV container vs bare PCM (`container=none`) |
| `encoding_mp3` / `encoding_flac` / `encoding_opus_ogg` / `encoding_mulaw_wav` / `encoding_aac` | each codec's container magic bytes |
| `sample_rate_48000` / `sample_rate_16000` | WAV sample rate matches the request |
| `bit_rate_mp3_32000` | `bit_rate` accepted for mp3 |
| `speed_duration` | synth at 0.7 / 1.0 / 1.5 — duration must shrink as speed rises (strongest signal) |
| `pronunciation_ipa` | inline IPA pronunciation override accepted + audio produced |
| `text_limit_exceeded` | text > 2000 chars → expects HTTP 413 (negative test) |
| `mip_opt_out` / `tag` | passthrough flags accepted (smoke) |

### `e2e/e2e_test_streaming.py` — websocket (bidirectional `/v1/speak`)

Streaming-specific behavior. Drives `tts_stress.py` headless
(`--no-playback --once --summary-jsonl`) and validates each connection's audio +
protocol acks from the summary JSON.

```bash
uv run e2e/e2e_test_streaming.py your-tts-endpoint --region us-east-2
uv run e2e/e2e_test_streaming.py --list
```

| Scenario | What it checks |
|---|---|
| `basic` / `concurrent_5` | `Speak`→audio + `Flushed`; 5-way concurrency |
| `multi_phrase_flush` | 3 phrases — multiple `Flushed` acks (Speak→Flush→Flushed loop) |
| `encoding_linear16_24k` | explicit linear16 @ 24 kHz — non-silent audio |
| `encoding_mulaw_8k` | streaming-supported companded codec (bytes-only check) |
| `speed_fast` | `speed=1.4` over the streaming transport (smoke) |
| `voice_alt` | alternate voice (PASS-WITH-NOTE if unbundled) |
| `mip_opt_out` | passthrough flag (smoke) |

Both drivers: exit code 0 = all pass; `tolerated_error_substring` scenarios
PASS-WITH-NOTE when the endpoint returns a known "not supported by this bundle"
error. Per-scenario logs + aggregated `results.json` land under
`/tmp/dg-sagemaker-e2e/tts-batch|tts-streaming/<timestamp>/`. Parameter coverage
is scoped to the TTS docs
(https://developers.deepgram.com/docs/tts-media-output-settings,
https://developers.deepgram.com/docs/tts-voice-controls) as of the June 2026 audit.

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

### No audio output

- Ensure the `--voice` is valid for the deployed TTS model — see the [Voices documentation](https://developers.deepgram.com/docs/tts-models)
- Check system volume and speaker settings
- Enable debug logging: `--log-level DEBUG`
