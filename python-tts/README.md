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
