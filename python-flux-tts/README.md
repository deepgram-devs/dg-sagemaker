# python-flux-tts — Deepgram Flux TTS on SageMaker

End-to-end and stress client for Deepgram Flux TTS (`/v2/speak`) deployed as a
SageMaker endpoint. Flux TTS differs from Aura-2 in ways that matter for testing:

| | Aura-2 | Flux TTS |
|---|---|---|
| path | `/v1/speak` | `/v2/speak` |
| model string | `aura-2-<voice>-en` | `flux-<voice>-en` |
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

## Voices

The model string is `flux-<voice>-en`; `flux-alexis-en` is the default used by
both batteries. The authoritative voice list for a given deployment is the one
its bundle ships — pass `--model` to select another.

## Metering

After any run, reconcile metering from the `deepgram-aws-sagemaker` repo:

```bash
AWS_PROFILE=shared-dev uv run --project deploy-script \
  deploy-script/test/audit_metering.py <endpoint> --minutes 30 --region us-east-2
```

Expect `Category=tts` with `consumed_units > 0`. At 45 units per 1k characters a
single short scenario rounds to 1–2 units, so run the concurrency scenarios if
you want a number that is unambiguously non-trivial.
