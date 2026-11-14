# python-flux-tts — Deepgram Flux TTS on SageMaker

End-to-end and stress client for Deepgram Flux TTS (`/v2/speak`) deployed as a
SageMaker endpoint. Flux TTS differs from Aura-2 in ways that matter for testing:

| | Aura-2 | Flux TTS |
|---|---|---|
| path | `/v1/speak` | `/v2/speak` |
| model string | `aura-2-<voice>-en` | `flux-<voice>-<lang>` (e.g. `flux-alexis-en`, `flux-aina-es`) |
| turn model | continuous | **turn-based** (`Speak` … `Flush`) |
| unknown query params | ignored | **rejected** with a 400 |
| billing tier | `aura-2` | `flux-tts`, 45 units per 1k characters |

Both transports are served off the **one** `streaming` image:

- **streaming** — WebSocket `/v2/speak`, via SageMaker bidirectional streaming.
- **batch** — `POST /invocations` with
  `x-amzn-sagemaker-custom-attributes: v2/speak?model=...` and a JSON
  `{"text": "..."}` body.

## Instance types

Flux TTS is single-GPU, but **g6 is the floor** — use `ml.g6.2xlarge` or
`ml.g6e.2xlarge`. It does not run on `ml.g5.*` or `ml.g4dn.*`. (This is narrower
than Flux ASR, which does run on g5.)

The listing ships English and the other languages as **separate versions**
(the version title lists the languages). The non-English version needs a
`*.4xlarge` host — `ml.g7.4xlarge` (recommended), `ml.g7e.4xlarge` or
`ml.g6e.4xlarge` — and only serves its own voices: pass `--voice` with one of
them (e.g. `--voice flux-aina-es`) when testing it, since the drivers default to
an English voice.

## Setup

```bash
cd python-flux-tts
uv sync
```

## Run

```bash
# streaming battery
AWS_PROFILE=shared-dev uv run e2e/e2e_test_streaming.py <endpoint> --region us-east-2

# batch battery
AWS_PROFILE=shared-dev uv run e2e/e2e_test_batch.py <endpoint> --region us-east-2

# list scenarios without running them
uv run e2e/e2e_test_streaming.py <endpoint> --list
```

Judge by the `PASS`/`FAIL` lines in the summary table, not by any "Errored"
count — the SageMaker bidirectional-streaming transport reports a teardown error
on essentially every connection, even on success.

## Output encodings

Both batteries cover every output encoding `/v2/speak` serves on their transport:

| transport | scenarios | check |
|---|---|---|
| batch | `encoding_mp3` / `encoding_opus` / `encoding_flac` | enough bytes, right container magic |
| batch | `encoding_aac` | enough bytes (AAC is a raw codec stream with no magic bytes) |
| batch | `encoding_mulaw_wav` / `encoding_alaw_wav` | 8 kHz, WAV container |
| both | `encoding_linear16_16k` | audible linear16 at 16 kHz |
| streaming | `encoding_mulaw_8k` / `encoding_alaw_8k` | enough bytes (companded, no header) |
| streaming | `compressed_encoding_rejected` | mp3/opus/aac/flac refused on the WebSocket |

The byte floor is the check that matters. A broken encoder returns HTTP 200
with an empty body, which no status-code check catches. mp3/opus/aac/flac
reject `sample_rate`, so those requests omit it.

## Speed and other voice controls are English-only

Flux TTS voice controls (`speed`, pronunciation) are available in English only;
the public [TTS voice controls](https://developers.deepgram.com/docs/tts-voice-controls)
page is the source of truth. A non-English voice still accepts a valid `speed`
(in range, on the 0.05 increment) but does not render faster: measured
2026-11-14 with `flux-aina-es`, the speed=1.5 render was 0.96× (streaming) and
0.99× (batch) the default duration, against 0.72× for `flux-alexis-en`.

So both batteries **skip** the `speed` scenario for a non-English `--voice`
(the row prints SKIP with the reason) instead of failing it. `configure_speed`
and `speed_out_of_range` still run in every language: they check that the
control is acknowledged and range-checked, which does not depend on language.

## Volume normalization

Both batteries run `normalize_volume`: one render with `normalize_volume=true`
and one with `normalize_volume=false` (the opt-out from the default loudness
normalization). Each must be accepted and return audible audio. The on/off RMS
ratio is printed in the notes but not gated, because whether normalization makes
a voice louder or quieter depends on that voice's starting loudness.

## Voices

The model string is `flux-<voice>-<lang>`; `flux-alexis-en` is the default used by
both batteries. The authoritative voice list for a given deployment is the one
its bundle ships — pass `--voice` to select another.

## Metering

After any run, reconcile metering from the `deepgram-aws-sagemaker` repo:

```bash
AWS_PROFILE=shared-dev uv run --project deploy-script \
  deploy-script/test/audit_metering.py <endpoint> --minutes 30 --region us-east-2
```

Expect `Category=tts` with `consumed_units > 0`. At 45 units per 1k characters a
single short scenario rounds to 1–2 units, so run the concurrency scenarios if
you want a number that is unambiguously non-trivial.
