# aws-deepgram-sagemaker

Automation scripts for testing Deepgram services running on Amazon SageMaker as an "Endpoint" resource.

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

## Flux (Conversational STT)

### Python

See [python-flux/README.md](python-flux/README.md) for full setup and usage.

Scripts:

- [`flux_stress.py`](python-flux/flux_stress.py) `file` — streams a WAV file to multiple Flux connections at real-time pace
- [`flux_stress.py`](python-flux/flux_stress.py) `microphone` — streams live microphone audio to multiple Flux connections
- [`flux_stress.py`](python-flux/flux_stress.py) `list-endpoints` — lists available SageMaker endpoints in the target region

End-to-end correctness gate ([`python-flux/e2e/`](python-flux/e2e/)) — Flux is streaming-only (`/v2/listen`), so a single driver covers it:

- [`e2e/e2e_test_streaming.py`](python-flux/e2e/e2e_test_streaming.py) — drives `flux_stress.py file` through basic / concurrency / connection-param / multilingual / in-band-control / negative scenarios, scoring each connection's combined `EndOfTurn` transcript against the reference by WER

