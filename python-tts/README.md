# Deepgram SageMaker Text-to-Speech Stress Test Client

A Python client for stress testing Deepgram Text-to-Speech (TTS) endpoints deployed on AWS SageMaker. This tool supports multiple concurrent streaming connections for load testing, with audio playback from a single selectable connection.

## Features

- **Multiple Concurrent Connections**: Create multiple simultaneous bidirectional streaming connections to a Deepgram TTS endpoint
- **Selective Audio Playback**: Route audio output from any single connection to local system speakers while other connections discard audio
- **Configurable Duration**: Specify how long the test should run
- **Voice Selection**: Choose any Deepgram TTS voice
- **Real-time Streaming**: Continuously stream text phrases to the endpoint while receiving synthesized audio
- **Graceful Shutdown**: Properly closes connections using the Deepgram Close message API
- **Detailed Logging**: Comprehensive debug and info logging for monitoring test execution

## Requirements

- Python 3.12 or higher
- AWS credentials configured (via AWS CLI, environment variables, or IAM role)
- PyAudio (for audio playback)
- On macOS: `brew install portaudio` (required for PyAudio)

## Installation

### 1. Install Python Dependencies

Using `uv` (recommended):

```bash
cd python-tts
uv sync
```

Or using `pip`:

```bash
cd python-tts
pip install -r requirements.txt
```

### 2. Configure AWS Credentials

Ensure your AWS credentials are configured:

```bash
aws configure
```

Or set environment variables:
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
```

### 3. Install PyAudio (macOS)

```bash
brew install portaudio
pip install pyaudio
```

## Usage

### Basic Usage

```bash
uv run tts_stress.py your-sagemaker-endpoint-name
```

### Full Example with Options

```bash
uv run tts_stress.py my-tts-endpoint \
  --connections 5 \
  --playback 2 \
  --duration 60 \
  --voice aura-2-thalia-en \
  --region us-east-2 \
  --log-level INFO
```

## Command-Line Options

### Required Arguments

- **`endpoint_name`**: Name of your SageMaker endpoint (required)

### Optional Arguments

- **`--connections`** (default: 1)
  - Number of simultaneous streaming connections to create
  - Example: `--connections 10` creates 10 connections

- **`--playback`** (default: 1)
  - Connection ID for audio playback to system speakers
  - Must be between 1 and the number of connections
  - Example: `--playback 3` plays audio from connection 3

- **`--duration`** (default: 30)
  - How long to run the test in seconds
  - Example: `--duration 120` runs the test for 2 minutes

- **`--voice`** (default: aura-2-thalia-en)
  - Deepgram TTS voice to use
  - Other examples: `aura-asteria-en`, `aura-luna-en`
  - Example: `--voice aura-asteria-en`

- **`--region`** (default: us-east-2)
  - AWS region where the SageMaker endpoint is deployed
  - Example: `--region us-west-2`

- **`--log-level`** (default: INFO)
  - Logging verbosity level
  - Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
  - Example: `--log-level DEBUG` for detailed debugging

## Examples

### Single Connection Test

Test a single TTS connection for 30 seconds:

```bash
uv run tts_stress.py my-endpoint --duration 30
```

### Multi-Connection Load Test

Create 5 concurrent connections with audio playback from connection 3:

```bash
uv run tts_stress.py my-endpoint --connections 5 --playback 3 --duration 60
```

### Extended Duration Test

Run a stress test for 5 minutes (300 seconds):

```bash
uv run tts_stress.py my-endpoint --connections 10 --duration 300
```

### Different Voice

Use a different voice for synthesis:

```bash
uv run tts_stress.py my-endpoint --voice aura-asteria-en --duration 60
```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
uv run tts_stress.py my-endpoint --log-level DEBUG --connections 2 --duration 30
```

## How It Works

1. **Initialization**: Creates multiple bidirectional streaming connections to the SageMaker endpoint
2. **Text Streaming**: Continuously sends test phrases (configurable in `TEST_PHRASES`) to all connections
3. **Audio Reception**: Each connection receives synthesized audio from the Deepgram TTS engine
4. **Selective Playback**: Audio from the selected connection is played to system speakers in real-time
5. **Graceful Shutdown**: After the specified duration:
   - Sends a Close message to all connections
   - Closes input streams (stops sending text)
   - Closes output streams
   - Waits for any remaining audio to be received and played
   - Exits cleanly

## Test Phrases

The script uses a rotating set of test phrases for TTS synthesis:

- "Hello world"
- "Testing text to speech"
- "Streaming audio data"
- "Multiple connections"
- "SageMaker Deepgram integration"

To customize the test phrases, edit the `TEST_PHRASES` list in `tts_stress.py`.

## Performance Metrics

The script outputs metrics for each connection:

- Number of phrases sent
- Total bytes of audio received
- Connection duration

## Troubleshooting

### Audio Not Playing

- Ensure `--playback` parameter specifies a valid connection ID (1 to N)
- Check system audio output settings
- Verify PyAudio is properly installed: `python -c "import pyaudio; print('OK')"`

### Connection Errors

- Verify the endpoint name is correct
- Ensure AWS credentials are configured
- Check that the endpoint is in the specified region
- Verify the endpoint is running and accepting connections
  - CloudWatch Logs for the SageMaker Endpoint can aid in identifying server-side errors

### Timeout Errors

- Increase `--duration` if the endpoint is slow
- Reduce `--connections` to lower load
- Check CloudWatch logs for the SageMaker endpoint

### No Audio Output

- Ensure you are using a valid Deepgram voice for the product you deployed; check the [Voices documentation](https://developers.deepgram.com/docs/tts-models)
- Verify PyAudio installation: `brew install portaudio && pip install pyaudio`
- Check system volume and speaker settings
- Enable debug logging: `--log-level DEBUG`
