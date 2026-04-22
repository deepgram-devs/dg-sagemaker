# Deepgram SageMaker STT — AWS SDK v2 Java Load Test

Java equivalent of [`python-stt/stt_wav_stress.py`](../../../python-stt) — streams a WAV file
to a Deepgram STT model on SageMaker using the AWS SDK v2 HTTP/2 bidirectional streaming API.

Designed to verify that the AWS SDK v2 Java client can sustain high concurrency (400+ bidi streams)
when configured with `maxStreams=1` (one stream per TCP connection).

## Prerequisites

- Java 17+ (tested with Amazon Corretto 17)
- AWS credentials configured (`aws configure` or env vars)

## Setup

```bash
# Install Java 17 if needed (macOS)
brew install --cask corretto@17

# Build
./gradlew build

# Build fat jar for easy distribution
./gradlew fatJar
```

## Usage

```bash
# Single connection (smoke test)
./gradlew run --args="my-deepgram-endpoint --file english.wav"

# 400 connections, batched 50 at a time with 2s delay between batches
./gradlew run --args="my-deepgram-endpoint --file english.wav \
    --connections 400 \
    --batch-size 50 \
    --batch-delay 2 \
    --region us-west-2"

# With fat jar
java -jar build/libs/dg-sagemaker-java-loadtest-1.0.0-all.jar \
    my-deepgram-endpoint --file english.wav --connections 400
```

## Key HTTP/2 Configuration

The critical setting is `--max-streams-per-connection` (default: 1):

```
--max-streams-per-connection 1    One bidi stream per TCP connection (RECOMMENDED)
--max-streams-per-connection 100  Default HTTP/2 multiplexing (causes starvation)
```

With `maxStreams=1`, the SDK opens a separate TCP connection for each bidi stream.
This prevents the HTTP/2 multiplexing starvation issue where audio chunks from
different streams queue behind each other, causing Deepgram's bounded queue to overflow.

Equivalent to `_MAX_STREAMS_PER_CONNECTION = 1` in the Python stress test.

### Recommended client configuration

```java
SdkAsyncHttpClient httpClient = NettyNioAsyncHttpClient.builder()
    .protocol(Protocol.HTTP2)
    .maxConcurrency(1000)
    .http2Configuration(
        Http2Configuration.builder()
            .maxStreams(1L)          // <-- one bidi stream per TCP connection
            .build()
    )
    .build();

SageMakerRuntimeHttp2AsyncClient sagemakerClient =
    SageMakerRuntimeHttp2AsyncClient.builder()
        .httpClient(httpClient)
        .build();
```

## Options

```
Usage: dg-sagemaker-loadtest ENDPOINT_NAME [OPTIONS]

      ENDPOINT_NAME           SageMaker endpoint name

Connection Control:
  -f, --file=FILE             Path to 16-bit PCM WAV file
  -c, --connections=N         Total simultaneous connections (default: 1)
      --batch-size=N          Connections per batch (default: 0 = all at once)
      --batch-delay=SECONDS   Delay between batches (default: 0)
      --loop / --no-loop      Loop audio file continuously (default: true)
      --duration=SECONDS      Stop after N seconds (default: 0 = until audio ends)

HTTP/2 Tuning:
      --max-streams-per-connection=N   HTTP/2 max streams per TCP connection (default: 1)
      --max-concurrency=N              Netty max concurrency (default: connections + 50)
      --connection-timeout=SECONDS     TCP connection timeout (default: 30)

Deepgram Model:
      --model=MODEL           Deepgram model (default: nova-3)
      --language=CODE         Language code (default: en)
      --punctuate             Enable punctuation (default: true)
      --diarize               Enable diarization (default: false)
      --interim-results       Enable partial results (default: false)
      --smart-format          Enable smart formatting (default: false)
      --keyterms=TERMS        Comma-separated keyterms (nova-3)

      --region=REGION         AWS region (default: us-west-2)
      --log-level=LEVEL       DEBUG|INFO|WARN|ERROR (default: INFO)
```

## Netty Tunable Parameters

Beyond the CLI options, the following `NettyNioAsyncHttpClient` parameters can be
tuned in the source code (`SageMakerStreamLoadTest.createClient()`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `maxStreams(Long)` | `1` | HTTP/2 streams per TCP connection. **Must be 1** for bidi streaming to avoid multiplexing starvation. |
| `maxConcurrency(Integer)` | `connections + 50` | Total in-flight requests across all connections in the pool. With `maxStreams=1`, this equals max TCP connections. |
| `connectionTimeout(Duration)` | `30s` | TCP connection establishment timeout. |
| `connectionAcquireTimeout(Duration)` | SDK default (~30s) | How long to wait for a connection from the pool before throwing `Acquire operation took longer`. Increase if seeing acquire timeouts under heavy load. |
| `connectionMaxIdleTime(Duration)` | SDK default | Close idle connections after this duration. Useful for long-running tests to avoid stale connections. |
| `healthCheckPingPeriod(Duration)` | `5s` | HTTP/2 PING frame interval for connection health checks. |
| `readTimeout(Duration)` | SDK default | Socket read timeout. May need increasing for slow models or large audio files. |
| `writeTimeout(Duration)` | SDK default | Socket write timeout. |

Example adding acquire timeout:
```java
NettyNioAsyncHttpClient.builder()
    .protocol(Protocol.HTTP2)
    .maxConcurrency(10)
    .connectionAcquireTimeout(Duration.ofSeconds(60))
    .http2Configuration(
        Http2Configuration.builder()
            .maxStreams(1L)
            .healthCheckPingPeriod(Duration.ofSeconds(5))
            .build()
    )
    .build();
```
