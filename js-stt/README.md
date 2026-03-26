## Node.js Transcription Load Test

To run the Node.js transcription load test:

1. Set your AWS credentials (e.g. `AWS_SHARED_CREDENTIALS_FILE` and `AWS_PROFILE` variables)
1. Ensure Node.js is installed
1. Set the AWS region, SageMaker Endpoint name, input audio file name, and query string parameters in `stt.file.ts`
1. Run `npx tsx stress-stt.ts`
