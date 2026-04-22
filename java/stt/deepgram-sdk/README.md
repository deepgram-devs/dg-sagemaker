# Deepgram SageMaker STT — Deepgram Java SDK Load Test

Streams a WAV file to a Deepgram STT model on SageMaker using the
[Deepgram Java SDK](https://github.com/deepgram/deepgram-java-sdk) plus the
[Deepgram SageMaker transport](https://github.com/deepgram/deepgram-java-sdk-transport-sagemaker).

SDK-level equivalent of the raw AWS SDK load test under [`../aws-sdk`](../aws-sdk).
Use this project when you want to validate the full SDK path that a typical
application takes in production:

```
DeepgramClient -> SageMakerTransportFactory -> SageMakerTransport -> SageMakerRuntimeHttp2AsyncClient
```

## Prerequisites

- Java 17+ (tested with Amazon Corretto 17)
- AWS credentials configured (`aws configure` or environment variables)

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
java -jar build/libs/dg-sdk-sagemaker-loadtest-1.0.0-all.jar \
    my-deepgram-endpoint --file english.wav --connections 400
```

## Options

```
Usage: dg-sdk-loadtest ENDPOINT_NAME [OPTIONS]

      ENDPOINT_NAME           SageMaker endpoint name

Connection Control:
  -f, --file=FILE             Path to 16-bit PCM WAV file
  -c, --connections=N         Total simultaneous connections (default: 1)
      --batch-size=N          Connections per batch (default: 0 = all at once)
      --batch-delay=SECONDS   Delay between batches (default: 0)
      --loop / --no-loop      Loop audio file continuously (default: true)
      --duration=SECONDS      Stop after N seconds (default: 0 = until audio ends)
      --max-retries=N         Max retries per connection on retryable errors (default: 10)

Deepgram Model:
      --model=MODEL           Deepgram model (default: nova-3)
      --interim-results       Enable partial results (default: false)

      --region=REGION         AWS region (default: us-west-2)
      --log-level=LEVEL       DEBUG|INFO|WARN|ERROR (default: INFO)
```

## HTTP/2 Configuration

The transport dependency ships with safe defaults for bidirectional streaming
(one HTTP/2 stream per TCP connection). If you need to tune Netty HTTP/2
parameters directly — `maxStreams`, `maxConcurrency`, timeouts, etc. — use the
AWS SDK project under [`../aws-sdk`](../aws-sdk), which drives
`NettyNioAsyncHttpClient` and `Http2Configuration` explicitly.

## Building the transport from source

The default build pulls `com.deepgram:deepgram-sagemaker` from Maven Central.
To test against an unreleased build of the transport:

1. Clone https://github.com/deepgram/deepgram-java-sdk-transport-sagemaker
2. Build the JAR and copy it to `libs/deepgram-sagemaker-transport.jar`
3. In [`build.gradle`](build.gradle), comment out the Maven coordinate line and
   uncomment the `implementation files('libs/...')` line below it.
