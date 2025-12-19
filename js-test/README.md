# JavaScript/TypeScript Implementation

A TypeScript project for testing Deepgram Text-to-Speech (TTS) using AWS SageMaker endpoints with bidirectional streaming.

## Prerequisites

- Node.js (v18 or higher recommended)
- Yarn or npm package manager
- AWS credentials configured (via AWS CLI, environment variables, or IAM role)
- Access to a deployed SageMaker endpoint with Deepgram TTS model

## Installation

1. Install all dependencies using Yarn:
   ```bash
   yarn install
   ```

   Alternatively, if you prefer npm:
   ```bash
   npm install
   ```

2. Install a TypeScript runner (if not already installed):
   ```bash
   yarn add -D tsx
   ```
   
   Or with npm:
   ```bash
   npm install -D tsx
   ```

## Configuration

Before running the scripts, you need to configure the following in `tts.ts`:

- **Region**: Update the `region` variable (line 17) to match your SageMaker endpoint region
- **Endpoint Name**: Update the `endpointName` in the `config` object (line 29) to your SageMaker endpoint name
- **Model**: Update the `modelQueryString` (line 21) to specify the Deepgram voice model you want to use
- **Input Text**: Modify the `inputText` variable (line 23) to change the text that will be converted to speech

## Running the Scripts

### Run the main TTS script

Test a single voice with the default configuration:

```bash
yarn tsx tts.ts
```

Or with npm:
```bash
npx tsx tts.ts
```

### Run the test-all-voices script

Test all available Deepgram aura-2 English voices:

```bash
yarn tsx test-all-voices.ts
```

Or with npm:
```bash
npx tsx test-all-voices.ts
```

## Scripts Overview

- **`tts.ts`**: Main script that invokes a SageMaker endpoint with bidirectional streaming to generate and play TTS audio for a single voice
- **`test-all-voices.ts`**: Test script that iterates through all Deepgram aura-2 English voices and plays audio for each one

## Notes

- The scripts use bidirectional streaming to send text chunks and receive audio responses in real-time
- Audio is played through your default audio device using the `@mastra/node-audio` package
- Make sure your AWS credentials are properly configured before running the scripts
- The scripts chunk text into 20-word segments and send them with flush messages to the SageMaker endpoint
- Audio is received in linear16 format at 24kHz sample rate (mono, 16-bit)

