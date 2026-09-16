# Invoking a Deepgram SageMaker endpoint

One rule covers every mode: **the Deepgram API path is mandatory and rides
alongside the request**, because SageMaker only knows `/invocations`.

| Mode | SageMaker API | Where the Deepgram path + query go | Endpoint type |
|---|---|---|---|
| Streaming | `InvokeEndpointWithBidirectionalStream` (HTTP/2, **port 8443**) | `ModelInvocationPath="v1/listen"`, `ModelQueryString="model=nova-3&language=en&…"` | real-time |
| Synchronous | `InvokeEndpoint` | `CustomAttributes="v1/listen?model=nova-3&language=en"` (header `X-Amzn-SageMaker-Custom-Attributes`) | real-time |
| Asynchronous | `InvokeEndpointAsync` | — | **temporarily not supported** for Marketplace-hosted Deepgram; contact a Deepgram representative |

Paths: `v1/listen` (Nova-3), `v2/listen` (Flux), `v1/speak` (Aura-2), `v2/speak` (Flux TTS).
Without a path the container returns 404.

## Required parameters per product

| Product | Path | Always | Streaming also | Never |
|---|---|---|---|---|
| Nova-3 monolingual | `v1/listen` | `model=nova-3`, `language=<code in the deployed version>` | `encoding=linear16&sample_rate=<hz>` (raw PCM) | — |
| Nova-3 multilingual | `v1/listen` | `model=nova-3`, **`language=multi`** | same | a specific language code |
| Flux English | `v2/listen` | `model=flux-general-en`, `encoding=linear16`, `sample_rate=<hz>` | — | `language=` |
| Flux multilingual | `v2/listen` | `model=flux-general-multi`, `encoding`, `sample_rate` | — | `language=` |
| Aura-2 | `v1/speak` | `model=aura-2-<voice>-<lang>` (e.g. `aura-2-thalia-en`) | `encoding=linear16&sample_rate=24000` recommended | — |
| Flux TTS | `v2/speak` | `model=flux-<voice>-<lang>` (e.g. `flux-alexis-en`) | `encoding=linear16&sample_rate=24000` | `speed`, unknown params (400) |

Other Deepgram features (`smart_format`, `punctuate`, `diarize`, `keyterm`,
`redact`, …) are ordinary query parameters. Unknown parameters are rejected with
400 rather than ignored.

## Synchronous — Python (boto3)

```python
import boto3, json
rt = boto3.client("sagemaker-runtime", region_name="us-east-2")
with open("audio.wav", "rb") as f:
    r = rt.invoke_endpoint(
        EndpointName="my-deepgram-stt",
        ContentType="audio/wav", Accept="application/json",
        CustomAttributes="v1/listen?model=nova-3&language=en&smart_format=true",
        Body=f.read())
print(json.loads(r["Body"].read())["results"]["channels"][0]["alternatives"][0]["transcript"])
```
Text-to-speech: `ContentType="application/json"`, `Body=json.dumps({"text": "…"})`,
`CustomAttributes="v1/speak?model=aura-2-thalia-en&encoding=linear16&sample_rate=24000"`;
the response body is audio.

CLI equivalent:
```bash
aws sagemaker-runtime invoke-endpoint --endpoint-name my-deepgram-stt --region us-east-2 \
  --content-type audio/wav --custom-attributes "v1/listen?model=nova-3&language=en" \
  --body fileb://audio.wav --cli-binary-format raw-in-base64-out out.json
```

## Asynchronous

Temporarily not supported for Marketplace-hosted Deepgram. For files over 25 MB
split the audio, or stream it faster than real time through a Streaming listing.
If you have an asynchronous use case, reach out to a Deepgram representative.

## Streaming — Python (`aws-sdk-sagemaker-runtime-http2` ≥ 0.11, `[awscrt]` extra)

```python
import asyncio, boto3
from aws_sdk_sagemaker_runtime_http2.client import AsyncSageMakerRuntimeHTTP2Client
from aws_sdk_sagemaker_runtime_http2.config import AsyncSageMakerRuntimeHTTP2Config
from aws_sdk_sagemaker_runtime_http2.models import (InvokeEndpointWithBidirectionalStreamInput,
                                                    RequestPayloadPart, RequestStreamEventPayloadPart,
                                                    ResponseStreamEventPayloadPart)
from smithy_http.aio.crt import AWSCRTHTTPClient

creds = boto3.Session().get_credentials().get_frozen_credentials()   # profile / SSO / role all work
client = AsyncSageMakerRuntimeHTTP2Client(config=AsyncSageMakerRuntimeHTTP2Config(
    region="us-east-2",
    endpoint_uri="https://runtime.sagemaker.us-east-2.amazonaws.com:8443",   # :8443 is mandatory
    aws_access_key_id=creds.access_key, aws_secret_access_key=creds.secret_key,
    aws_session_token=creds.token,
    transport=AWSCRTHTTPClient(),                                            # HTTP/2-capable transport
))

async def main():
    stream = await client.invoke_endpoint_with_bidirectional_stream(
        InvokeEndpointWithBidirectionalStreamInput(
            endpoint_name="my-deepgram-stt", model_invocation_path="v1/listen",
            model_query_string="model=nova-3&language=en&encoding=linear16&sample_rate=16000"))
    _, output = await stream.await_output()

    async def send(b: bytes, data_type: str):          # audio: "BINARY"; JSON control: "UTF8"
        await stream.input_stream.send(RequestStreamEventPayloadPart(
            value=RequestPayloadPart(bytes_=b, data_type=data_type)))

    # … send 100 ms PCM chunks with send(chunk, "BINARY"); optionally {"type":"KeepAlive"} as UTF8
    await send(b'{"type":"CloseStream"}', "UTF8")
    while (event := await output.receive()) is not None:
        if isinstance(event, ResponseStreamEventPayloadPart):
            print(event.value.bytes_.decode())          # Deepgram JSON messages
        else:                                            # ModelStreamError / InternalStreamFailure
            print("stream error:", event.value)
    await client.close()

asyncio.run(main())
```
Versions before 0.11 exposed `SageMakerRuntimeHTTP2Client` + `Config`; the
repository's drivers (`python-stt`, `python-flux`, `python-tts`,
`python-flux-tts`) show the 0.11 shape with a client plugin, which keeps
construction synchronous. Both generations surface a pre-upgrade rejection as
a 424 `ModelError`.

Messages: Nova-3 → `Results` with `channel.alternatives[0].transcript`, `is_final`,
`speech_final`. Flux → `TurnInfo` with `event` ∈ `Update | StartOfTurn |
EagerEndOfTurn | EndOfTurn` and `transcript`. TTS → binary audio frames plus
JSON `Metadata`, `Flushed`, `SpeechMetadata` (Flux TTS); send `{"type":"Speak","text":…}`,
`{"type":"Flush"}`, `{"type":"Close"}`.

TypeScript: `@aws-sdk/client-sagemaker-runtime-http2` with
`endpoint: "https://runtime.sagemaker.<region>.amazonaws.com:8443"`; the `Body`
is an async iterable of `{ PayloadPart: { Bytes, DataType: "BINARY" | "UTF8" } }`.
Java: the Deepgram Java SDK's SageMaker transport (`com.deepgram:deepgram-sagemaker`),
`.apiKey("unused")` — auth is SigV4.

Full runnable clients (file, microphone, load, e2e gates) live in this repository:
`python-stt/`, `python-flux/`, `python-tts/`, `python-flux-tts/`, `js-stt/`, `java/`.

## FIPS

Per-client `use_fips_endpoint=True` (boto3 `Config`) → `runtime-fips.sagemaker.<region>.amazonaws.com`.
Streaming keeps `:8443` on the FIPS host. Do **not** set `AWS_USE_FIPS_ENDPOINT`
process-wide with an SSO profile (the SSO portal has no FIPS hostname). Regions with FIPS: us-east-1, us-east-2, us-west-1,
us-west-2, ca-central-1, GovCloud.

## Behaviour to expect

- Streaming open fails with HTTP 424 `Failed to establish WebSocket connection`
  (within seconds, before any audio is sent) → the container rejected the
  request before the upgrade (bad model/language/path). SageMaker does not
  forward the container's own 400 text to streaming clients; read it in the
  endpoint's CloudWatch log. A client that instead hangs with no error is
  almost always missing `:8443` in the URI.
- Streaming listing hit with `InvokeEndpoint` → 400 `No such model/language/tier`.
- Latency metrics in CloudWatch are in **microseconds**; `ModelLatency` is not
  emitted for streaming — use `FirstChunkLatency` and `ConcurrentRequestsPerModel`.
