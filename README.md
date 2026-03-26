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

---

## Text-to-Speech (TTS)

### JavaScript

TBD

### Python

See [python-tts/README.md](python-tts/README.md) for full setup and usage.

Scripts:
- `tts_stress.py` — streams text phrases to multiple simultaneous bidirectional connections; plays audio from one selectable connection

---

## Flux (Conversational STT)

### Python

See [python-flux/README.md](python-flux/README.md) for full setup and usage.

Scripts:

- `flux_stress.py file` — streams a WAV file to multiple Flux connections at real-time pace
- `flux_stress.py microphone` — streams live microphone audio to multiple Flux connections
- `flux_stress.py list-endpoints` — lists available SageMaker endpoints in the target region

