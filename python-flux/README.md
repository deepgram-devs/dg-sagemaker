# Deepgram Flux STT SageMaker Stress Test

Streams audio to multiple simultaneous bidirectional connections to a
**Deepgram Flux** model deployed on Amazon SageMaker. Two input modes are
supported:

- **`file`** — streams a WAV file at real-time pace (repeatable load testing).
- **`microphone`** — captures live audio from a microphone via PyAudio.

## What is Flux?

Flux (`flux-general-en`) is Deepgram's conversational speech recognition model
built for voice agents. Unlike Nova-3, Flux uses the **`/v2/listen` endpoint**
and a **turn-based protocol** with integrated end-of-turn detection — no
external VAD or server-side VAD configuration required.

Key differences from Nova-3:

| Aspect | Nova-3 | Flux |
|---|---|---|
| Endpoint path | `/v1/listen` | `/v2/listen` |
| Response type | `channel.alternatives` | `TurnInfo` events |
| Turn detection | External | Model-integrated |
| Barge-in detection | Manual | Native (`StartOfTurn`) |
| Dynamic configuration | No | `Configure` message |

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- AWS credentials configured (CLI, environment variables, or IAM role)
- A SageMaker endpoint running the Deepgram Flux model
  - The endpoint's engine config must include `listen_v2 = true`
- **For `file` mode:** a 16-bit PCM WAV file
- **For `microphone` mode:** PortAudio library (see [Setup](#setup))

### Convert audio to the required format

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 input.wav
```

## Setup

```bash
cd python-flux
uv sync
```

**macOS — microphone support requires PortAudio:**

```bash
brew install portaudio
uv sync
```

## Usage

```
uv run flux_stress.py <SUBCOMMAND> <endpoint_name> [options]
```

Subcommands:

| Subcommand | Description |
|---|---|
| `list-endpoints` | List available SageMaker endpoints in the target region |
| `file` | Stream a WAV audio file at real-time pace |
| `microphone` | Capture live microphone input and stream it in real-time |

Run `uv run flux_stress.py <subcommand> --help` for full option details.

---

## `list-endpoints` subcommand

Lists SageMaker endpoints in the target region along with their status and
timestamps. Useful for discovering available endpoint names before running a
stress test.

### Options

| Flag | Default | Description |
|---|---|---|
| `--region REGION` | `us-east-1` | AWS region to query |
| `--status STATUS` | *(all)* | Filter by status: `inservice`, `creating`, `updating`, `rollingback`, `systemupdating`, `failed`, `deleting`, `outofservice` |
| `--log-level LEVEL` | `WARNING` | Logging verbosity |

### Examples

List all endpoints in the default region:

```bash
uv run flux_stress.py list-endpoints
```

List only `InService` endpoints in a specific region:

```bash
uv run flux_stress.py list-endpoints --region us-west-2 --status inservice
```

### Example output

```
SageMaker Endpoints  [us-east-1]
------------------------------------------------------------
ENDPOINT NAME            STATUS     CREATED               LAST MODIFIED
------------------------------------------------------------
my-flux-endpoint         InService  2026-03-01 14:22:05   2026-03-05 09:11:42
my-flux-endpoint-dev     Creating   2026-03-10 08:00:13   2026-03-10 08:00:13
------------------------------------------------------------
2 endpoint(s)
```

---

## `file` subcommand

### Options

| Flag | Default | Description |
|---|---|---|
| `--file WAV_FILE` | *(required)* | 16-bit PCM WAV file to stream |
| `--connections N` | `1` | Number of simultaneous Flux connections |
| `--model MODEL` | `flux-general-en` | Flux model variant |
| `--eot-threshold 0.5-0.9` | `0.7` (server default) | EndOfTurn confidence threshold |
| `--eager-eot-threshold 0.3-0.9` | *(disabled)* | Enables EagerEndOfTurn events; must be ≤ `--eot-threshold` |
| `--eot-timeout-ms 500-10000` | `5000` (server default) | Max silence before forced EndOfTurn |
| `--keyterms TERM1,TERM2` | *(none)* | Comma-separated keyterms for recognition boosting |
| `--region REGION` | `us-east-1` | AWS region |
| `--loop` | *(off)* | Loop the WAV file continuously |
| `--duration SECONDS` | *(until file ends)* | Stop automatically after N seconds |
| `--log-level LEVEL` | `INFO` | DEBUG / INFO / WARNING / ERROR / CRITICAL |

### Examples

Single connection, one pass through the file:

```bash
uv run flux_stress.py file my-flux-endpoint --file audio.wav
```

10 connections looping for 60 seconds:

```bash
uv run flux_stress.py file my-flux-endpoint \
  --file audio.wav \
  --connections 10 \
  --loop \
  --duration 60
```

Custom end-of-turn thresholds with eager detection:

```bash
uv run flux_stress.py file my-flux-endpoint \
  --file audio.wav \
  --eot-threshold 0.8 \
  --eager-eot-threshold 0.5 \
  --eot-timeout-ms 3000
```

Keyterm boosting with debug logging:

```bash
uv run flux_stress.py file my-flux-endpoint \
  --file audio.wav \
  --keyterms "SageMaker,Deepgram,Flux" \
  --log-level DEBUG
```

---

## `microphone` subcommand

### Options

| Flag | Default | Description |
|---|---|---|
| `--connections N` | `1` | Number of simultaneous Flux connections |
| `--model MODEL` | `flux-general-en` | Flux model variant |
| `--sample-rate HZ` | `16000` | Microphone sample rate |
| `--device INDEX` | system default | PyAudio input device index |
| `--list-devices` | — | List available input devices and exit |
| `--eot-threshold 0.5-0.9` | `0.7` (server default) | EndOfTurn confidence threshold |
| `--eager-eot-threshold 0.3-0.9` | *(disabled)* | EagerEndOfTurn threshold; must be ≤ `--eot-threshold` |
| `--eot-timeout-ms 500-10000` | `5000` (server default) | Max silence before forced EndOfTurn |
| `--keyterms TERM1,TERM2` | *(none)* | Comma-separated keyterms for recognition boosting |
| `--region REGION` | `us-east-1` | AWS region |
| `--duration SECONDS` | *(until Ctrl+C)* | Stop automatically after N seconds |
| `--log-level LEVEL` | `INFO` | DEBUG / INFO / WARNING / ERROR / CRITICAL |

### Examples

List available microphone input devices:

```bash
uv run flux_stress.py microphone my-flux-endpoint --list-devices
```

Single connection using the system default microphone:

```bash
uv run flux_stress.py microphone my-flux-endpoint
```

5 connections for 30 seconds:

```bash
uv run flux_stress.py microphone my-flux-endpoint \
  --connections 5 \
  --duration 30
```

Specific device with eager end-of-turn detection:

```bash
uv run flux_stress.py microphone my-flux-endpoint \
  --device 2 \
  --eot-threshold 0.8 \
  --eager-eot-threshold 0.5
```

---

## Output format

```
[Conn 1]   hello how are you [update]                        ← Update (interim, ~250ms)
[Conn 1] ~ hello, how are you? (87.3%) [eager, turn 0]      ← EagerEndOfTurn
[Conn 1] ✓ hello, how are you? (91.2%) [turn 0]             ← EndOfTurn (final)
[Conn 1]   ... resumed [turn 1]                              ← TurnResumed (barge-in after eager)
```

Legend:
- `✓` — `EndOfTurn` (final transcript)
- `~` — `EagerEndOfTurn` (high likelihood turn is complete; useful for speculative LLM pre-processing)
- `[update]` — `Update` (periodic interim transcript, not a turn boundary)
- `... resumed` — `TurnResumed` (user continued speaking after `EagerEndOfTurn`)

## Flux Protocol Summary

### Client → Server messages

| Message | Format | Purpose |
|---|---|---|
| Audio | Binary bytes | Raw PCM audio (80ms chunks recommended) |
| `Configure` | JSON text | Update thresholds or keyterms mid-stream |
| `KeepAlive` | JSON text | Prevent idle timeout |
| `Finalize` | JSON text | Flush buffered audio; force end current turn |
| `CloseStream` | JSON text | Gracefully terminate the stream |

### Server → Client messages

| Message | Description |
|---|---|
| `Connected` | Emitted once on stream open |
| `TurnInfo` | Transcript update; `event` field indicates state (see below) |
| `ConfigureSuccess` | Confirms a `Configure` was applied |
| `ConfigureFailure` | Rejects a `Configure` (e.g. constraint violation) |
| `Error` | Fatal server error; connection terminated |

### TurnInfo event states

```
[Ready]
    │ StartOfTurn
    ▼
[Speaking]
    │ EagerEndOfTurn (if eager_eot_threshold set)
    ▼
[AwaitingEnd]
   / \
TurnResumed  EndOfTurn
   │              │
[Speaking]   [Ready, turn_index++]
```

## Self-Hosted / SageMaker Notes

- The SageMaker endpoint engine configuration must include `listen_v2 = true`
- Flux must run on dedicated GPU resources isolated from other models
- The bidirectional stream uses HTTP/2 on port 8443:
  `https://runtime.sagemaker.<region>.amazonaws.com:8443`
