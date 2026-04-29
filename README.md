# aws-deepgram-sagemaker

Automation scripts for testing Deepgram services running on Amazon SageMaker as an "Endpoint" resource.

## Speech-to-Text (STT)

### JavaScript

TBD

### Python

See [python-stt/README.md](python-stt/README.md) for full setup and usage.

Scripts:

- `stt_microphone_stress.py` — streams live microphone audio; supports multiple simultaneous connections
- `stt_wav_stress.py stream` — streams a WAV file at real-time pace; repeatable load testing without a microphone
- `stt_wav_stress.py batch` — posts WAV files via HTTP with configurable concurrency; reports latency and throughput
- `stt_sdk.py` — single WAV streaming smoke test using the Deepgram Python SDK plus the SageMaker transport

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
- `tts_stress.py` — streams text phrases to multiple simultaneous bidirectional connections; plays audio from one selectable connection
- `tts_http.py` — sends one non-WebSocket SageMaker `InvokeEndpoint` HTTP request to `/v1/speak`
- `tts_sdk.py` — single text-to-speech streaming smoke test using the Deepgram Python SDK plus the SageMaker transport

---

## Flux (Conversational STT)

### Python

See [python-flux/README.md](python-flux/README.md) for full setup and usage.

Scripts:

- `flux_stress.py file` — streams a WAV file to multiple Flux connections at real-time pace
- `flux_stress.py microphone` — streams live microphone audio to multiple Flux connections
- `flux_stress.py list-endpoints` — lists available SageMaker endpoints in the target region
- `flux_sdk.py` — single WAV streaming smoke test using the Deepgram Python SDK `listen.v2` client plus the SageMaker transport
