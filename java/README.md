# Java clients for Deepgram on SageMaker

Java load test clients for Deepgram models deployed to an Amazon SageMaker endpoint. Each subfolder is a standalone Gradle project.

## Speech-to-Text (STT)

| Project | Description |
|---|---|
| [`stt/aws-sdk`](stt/aws-sdk) | Drives the SageMaker bidi streaming API directly via AWS SDK v2. Useful for isolating SDK/transport behavior from the Deepgram Java SDK. |
| [`stt/deepgram-sdk`](stt/deepgram-sdk) | Drives the same API via the Deepgram Java SDK plus the [Deepgram SageMaker transport](https://github.com/deepgram/deepgram-java-sdk-transport-sagemaker). Mirrors how production applications consume the service. |

Both projects are functionally equivalent to [`python-stt/stt_wav_stress.py stream`](../python-stt) — they stream a WAV file to a SageMaker endpoint at real-time pace, across one or many concurrent bidi connections, and report per-connection metrics.
