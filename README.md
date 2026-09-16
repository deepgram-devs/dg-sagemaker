# aws-deepgram-sagemaker

Automation scripts for testing Deepgram services running on Amazon SageMaker as an "Endpoint" resource.

## Agent-assisted setup (skills)

[`skills/deepgram-sagemaker/`](skills/deepgram-sagemaker/) is an installable agent
skill that walks a customer from AWS Marketplace subscription to a tested
SageMaker endpoint, running every AWS step through a deterministic script
instead of hand-typed console or CLI work. It follows the open
[`SKILL.md`](https://agentskills.io) format, so it works in Claude Code, Codex,
Cursor and other agents that read skills.

Install into a project:

```bash
npx skills add deepgram-devs/dg-sagemaker          # any SKILL.md-aware agent
# Claude Code plugin:
/plugin marketplace add deepgram-devs/dg-sagemaker
/plugin install deepgram-sagemaker@deepgram
```

Then ask the agent, e.g. "set up Deepgram Nova-3 streaming on SageMaker in
us-east-2". It needs AWS credentials for the target account and
[`uv`](https://docs.astral.sh/uv/) (each script declares its own dependencies).
The agent confirms with you before anything that costs money (subscribing,
creating an endpoint) or deletes resources, recommends an instance *pool*
rather than a single type, and refuses asynchronous endpoints, which are
temporarily not supported for Marketplace-hosted Deepgram (ask a Deepgram
representative if you need them). For capacity planning it will tell you to
measure concurrency on your own endpoint — or ask a Deepgram representative —
rather than quote a number.

The scripts also work on their own (`uv run skills/deepgram-sagemaker/scripts/<script>.py --help`):

- [`preflight.py`](skills/deepgram-sagemaker/scripts/preflight.py) — credentials, region, tool versions, and a permission smoke test per phase
- [`list_products.py`](skills/deepgram-sagemaker/scripts/list_products.py) — Deepgram's SageMaker listings and this account's subscription state (ACTIVE agreements only)
- [`subscribe.py`](skills/deepgram-sagemaker/scripts/subscribe.py) — the Marketplace Agreement API subscribe flow; quotes without `--accept`, subscribes with it
- [`resolve_model_package_arn.py`](skills/deepgram-sagemaker/scripts/resolve_model_package_arn.py) — product + version → per-region ModelPackage ARN, recommended and supported instance types
- [`check_quota.py`](skills/deepgram-sagemaker/scripts/check_quota.py) — per-type endpoint quota, current usage, in-flight requests; `--request N` opens an increase
- [`create_execution_role.py`](skills/deepgram-sagemaker/scripts/create_execution_role.py) — idempotent SageMaker execution role (+ S3 grant for async)
- [`deploy_endpoint.py`](skills/deepgram-sagemaker/scripts/deploy_endpoint.py) — Model + EndpointConfig + Endpoint with the required settings baked in; waits and diagnoses failures
- [`endpoint_status.py`](skills/deepgram-sagemaker/scripts/endpoint_status.py) — status, variant, log tail, named cause + next step
- [`invoke_test.py`](skills/deepgram-sagemaker/scripts/invoke_test.py) — one real streaming or synchronous request with the right path and params; explains the classic 400s
- [`configure_autoscaling.py`](skills/deepgram-sagemaker/scripts/configure_autoscaling.py) — target-tracking auto-scaling for real-time endpoints
- [`update_endpoint.py`](skills/deepgram-sagemaker/scripts/update_endpoint.py) — in-place update (instance type/count, AMI, env, model version) with rollback
- [`teardown_endpoint.py`](skills/deepgram-sagemaker/scripts/teardown_endpoint.py) — delete endpoint + config + model, then verify nothing is left

Reference material the skill reads lives in [`skills/deepgram-sagemaker/references/`](skills/deepgram-sagemaker/references/)
(`products.json` is the machine-readable catalog of listings, API paths, required parameters and instance types).

## Speech-to-Text (STT)

### JavaScript

See [js-stt/README.md](js-stt/README.md) for setup and usage. Built on the AWS SDK HTTP/2 bidirectional streaming client (`@aws-sdk/client-sagemaker-runtime-http2`); configuration (region, endpoint name, input file, query string) is edited inline at the top of each script.

Scripts:

- [`stt.file.ts`](js-stt/stt.file.ts) — streams a WAV file to a bidirectional streaming endpoint, chunking the file with keepalives
- [`stt.microphone.ts`](js-stt/stt.microphone.ts) — captures live microphone input and streams it to a bidirectional streaming endpoint
- [`stress-stt.ts`](js-stt/stress-stt.ts) — fires N parallel `stt.file.ts` invocations and reports success/failure counts and timing

### Python

See [python-stt/README.md](python-stt/README.md) for full setup and usage.

Scripts:

- [`stt_microphone_stress.py`](python-stt/stt_microphone_stress.py) — streams live microphone audio; supports multiple simultaneous connections
- [`stt_wav_stress.py`](python-stt/stt_wav_stress.py) `stream` — streams a WAV file at real-time pace; repeatable load testing without a microphone
- [`stt_wav_stress.py`](python-stt/stt_wav_stress.py) `batch` — posts WAV files via HTTP with configurable concurrency; reports latency and throughput
- [`stt_wav_async.py`](python-stt/stt_wav_async.py) — transcribes a WAV file (up to 1 GiB) via the SageMaker `InvokeEndpointAsync` API with S3 input/output; suits long-form audio beyond the synchronous invoke limit, with configurable concurrency and a latency/throughput summary

End-to-end correctness gates ([`python-stt/e2e/`](python-stt/e2e/)) — wrap the stress scripts and score each connection's transcript against a known reference (`spacewalk.wav`) via Word Error Rate; intended as the promotion gate before an endpoint goes live:

- [`e2e/e2e_test_streaming.py`](python-stt/e2e/e2e_test_streaming.py) — drives `stt_wav_stress.py stream` through ~10 scenarios (basic short/long-form, sustained + ramped concurrency, the major feature flags, an adversarial WebSocket-close path) and checks each connection's combined final transcript by WER
- [`e2e/e2e_test_batch.py`](python-stt/e2e/e2e_test_batch.py) — `--mode sync` (25 s sample via `invoke_endpoint`, ≤ 25 MB) or `--mode async` (~15 min / ~76 MB via `invoke_endpoint_async` + S3, incl. summarize); validates every returned transcript by WER

### Java

See [java/README.md](java/README.md) for an index of Java projects.

- [`java/stt/aws-sdk`](java/stt/aws-sdk) — WAV streaming load test built directly on AWS SDK v2 HTTP/2 bidi streaming
- [`java/stt/deepgram-sdk`](java/stt/deepgram-sdk) — same load test, via the Deepgram Java SDK + SageMaker transport

---

## Text-to-Speech (TTS)

### JavaScript

TBD

### Python

See [python-tts/README.md](python-tts/README.md) for full setup and usage.

Scripts:
- [`tts_stress.py`](python-tts/tts_stress.py) — streams text phrases to multiple simultaneous bidirectional connections; plays audio from one selectable connection

End-to-end correctness gates ([`python-tts/e2e/`](python-tts/e2e/)) — validate the **synthesized audio itself** (non-empty, correct container/codec, non-silent, requested sample rate, speed→duration), so no second transcription endpoint is required:

- [`e2e/e2e_test_batch.py`](python-tts/e2e/e2e_test_batch.py) — synchronous `invoke_endpoint` against `/v1/speak`; carries the full parameter matrix (model/encoding/sample_rate/bit_rate/container/speed, inline IPA override, 2000-char limit)
- [`e2e/e2e_test_streaming.py`](python-tts/e2e/e2e_test_streaming.py) — websocket `/v1/speak`; the streaming-only behaviors (`Speak`→audio, `Flush`→`Flushed`, sustained concurrency, streaming encodings, voice/speed)

---

## Flux TTS (Turn-based TTS)

### Python

See [python-flux-tts/README.md](python-flux-tts/README.md) for full setup and usage.
Flux TTS uses `/v2/speak` and a turn-based protocol (`Speak` … `Flush`), so it has
its own client rather than sharing the Aura-2 one. Requires `ml.g6.2xlarge` or
newer — it does not run on g5 or g4dn.

Scripts:

- [`flux_tts_client.py`](python-flux-tts/flux_tts_client.py) — shared client for both surfaces: `FluxTtsStream` (websocket `/v2/speak` over SageMaker bidirectional streaming) and `invoke_batch()` (`POST /invocations`)

End-to-end correctness gates ([`python-flux-tts/e2e/`](python-flux-tts/e2e/)) — both transports are served by the same endpoint, so one deployment covers both drivers:

- [`e2e/e2e_test_batch.py`](python-flux-tts/e2e/e2e_test_batch.py) — `POST /invocations` → `/v2/speak`; audio validity, `container=wav`, speed→duration, and negative controls (unknown param, Aura model rejected)
- [`e2e/e2e_test_streaming.py`](python-flux-tts/e2e/e2e_test_streaming.py) — websocket `/v2/speak`; the turn-based behaviors (per-turn `SpeechMetadata` accounting, multi-turn, incremental `Speak`, `Interrupt`, mid-stream `Configure{speed}`, concurrency)

---

## Flux (Conversational STT)

### Python

See [python-flux/README.md](python-flux/README.md) for full setup and usage.

Scripts:

- [`flux_stress.py`](python-flux/flux_stress.py) `file` — streams a WAV file to multiple Flux connections at real-time pace
- [`flux_stress.py`](python-flux/flux_stress.py) `microphone` — streams live microphone audio to multiple Flux connections
- [`flux_stress.py`](python-flux/flux_stress.py) `list-endpoints` — lists available SageMaker endpoints in the target region

End-to-end correctness gate ([`python-flux/e2e/`](python-flux/e2e/)) — Flux is streaming-only (`/v2/listen`), so a single driver covers it:

- [`e2e/e2e_test_streaming.py`](python-flux/e2e/e2e_test_streaming.py) — drives `flux_stress.py file` through basic / concurrency / connection-param / multilingual / in-band-control / negative scenarios, scoring each connection's combined `EndOfTurn` transcript against the reference by WER

