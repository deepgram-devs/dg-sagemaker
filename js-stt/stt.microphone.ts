// Sample script to capture microphone input and stream to Amazon SageMaker bidirectional 
// streaming endpoint with Deepgram transcription Voice AI models.
import {
    SageMakerRuntimeHTTP2Client,
    InvokeEndpointWithBidirectionalStreamCommand,
    InvokeEndpointWithBidirectionalStreamCommandOutput,
    RequestStreamEvent,
    InvokeEndpointWithBidirectionalStreamCommandInput
} from '@aws-sdk/client-sagemaker-runtime-http2';
import { getMicrophoneStream } from '@mastra/node-audio';
import { Readable } from 'stream';

// Configuration interface
interface Config {
    endpointName: string;
}

// Script Configuration

// CHANGE ME: Specify the region where your SageMake Endpoint is deployed.
const region: string = "us-west-2";
// CHANGE ME: This must correspond to the AWS region where your Endpoint is deployed.
const bidiEndpoint: string = `https://runtime.sagemaker.${region}.amazonaws.com:8443`;
// The internal WebSocket API route you want to access, used by Deepgram specifically.
const modelInvocationPath = 'v1/listen';
// CHANGE ME: Update this to the model parameters you want. Preview only supports nova-3, entity detection, and diarization.
const modelQueryString = 'model=nova-3&language=es';

// CHANGE ME: Update this value to the name of the SageMaker Endpoint you deploy
const config: Config = {
    endpointName: `2026-01-05-auto-scaling-test-trevor-sullivan`,
};

const sagemakerRuntimeClientForBidi = new SageMakerRuntimeHTTP2Client({
    region: region,
    endpoint: bidiEndpoint
});

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms)).then(() => console.log("Sleeping completed!"));

// Generator function that yields audio chunks from the local microphone
async function* audioStream(chunkSize: number = 1024*128) : AsyncIterable<RequestStreamEvent> {
    const KEEPALIVE_INTERVAL = 3000; // 3 seconds
    let lastKeepaliveTime = Date.now();

    let microphone = getMicrophoneStream();

    let streamActive = true;
    // Set up keepalive interval
    const keepaliveInterval = setInterval(() => {
        if (streamActive && Date.now() - lastKeepaliveTime >= KEEPALIVE_INTERVAL) {
            console.log('Sending keepalive message...');
            lastKeepaliveTime = Date.now();
        }
    }, KEEPALIVE_INTERVAL);
    
    try {
        for await (const chunk of microphone) {
            yield {
                PayloadPart: {
                    Bytes: new Uint8Array(chunk),
                    DataType: "BINARY",
                },
            };
        }
        console.log('Audio streaming complete. Continuing to send keepalive messages...');
        const keepaliveEndTime = Date.now() + 120000; // Keep alive for 40 seconds
        
        while (Date.now() < keepaliveEndTime) {
            const now = Date.now();
            if (now - lastKeepaliveTime >= KEEPALIVE_INTERVAL) {
                const timestamp = new Date(now).toISOString();
                console.log(`Sending post-stream keepalive message at ${timestamp}...`);
                yield {
                    PayloadPart: {
                        Bytes: new TextEncoder().encode(JSON.stringify({
                            type: "KeepAlive",
                        })),
                        DataType: "UTF8",
                    },
                };
                lastKeepaliveTime = now;
            }
            // Small sleep to prevent tight loop
            await sleep(3000);
        }

        // Close the connection after receiving all the response messages
        yield {
            PayloadPart: {
                Bytes: new TextEncoder().encode(JSON.stringify({
                    type: "CloseStream",
                })),
                DataType: "UTF8",
            },
        };
        
        console.log('Keepalive period completed.');
    } finally {
        streamActive = false;
        clearInterval(keepaliveInterval);
    }
}

// Invokes the Amazon SageMaker bidirectional stream API and processes response payloads
async function invokeEndpointWithBidirectionalStream(): Promise<InvokeEndpointWithBidirectionalStreamCommandOutput> {
    console.log('Invoking endpoint with bidirectional stream...');

    const invokeParams: InvokeEndpointWithBidirectionalStreamCommandInput = {
        EndpointName: config.endpointName,
        // Body: createRequestStream(), // as AsyncIterable<RequestStreamEvent>
        Body: audioStream(), // as AsyncIterable<RequestStreamEvent>
        ModelInvocationPath: modelInvocationPath,
        ModelQueryString: modelQueryString,
    };

    try {
        console.log('Using custom bidi endpoint:', bidiEndpoint);
        const command = new InvokeEndpointWithBidirectionalStreamCommand(invokeParams);
        console.log('Sending bidirectional stream request...');
        const response: InvokeEndpointWithBidirectionalStreamCommandOutput = await sagemakerRuntimeClientForBidi.send(command);

        console.log('Bidirectional stream response received. Processing...');
        console.log('Response metadata:', response.$metadata);

        if (response.Body) {
            let chunkCount = 0;
            const timeout = setTimeout(() => {
                console.log('Timeout waiting for bidirectional stream chunks');
            }, 20000); // 10 second timeout

            try {
                // Read responses from the bidirectional stream
                for await (const chunk of response.Body) {
                    chunkCount++;
                    console.log(`Processing bidirectional chunk ${chunkCount}:`, Object.keys(chunk));

                    if (chunk.PayloadPart && chunk.PayloadPart.Bytes) {
                        const chunkData = new TextDecoder().decode(chunk.PayloadPart.Bytes);
                        console.log('Bidirectional chunk data:', chunkData);
                        // console.log('Bidirectional chunk:', chunk.PayloadPart);
                    }

                    if (chunk.InternalStreamFailure) {
                        console.error('Bidirectional internal stream failure:', chunk.InternalStreamFailure);
                        break;
                    }

                    if (chunk.ModelStreamError) {
                        console.error('Bidirectional model stream error:', chunk.ModelStreamError);
                        break;
                    }
                }
                clearTimeout(timeout);
                console.log(`Processed ${chunkCount} bidirectional chunks total`);
            } catch (streamError) {
                clearTimeout(timeout);
                console.error('Error processing bidirectional stream:', streamError);
                throw streamError;
            }
        } else {
            console.log('No bidirectional response body received');
        }

        console.log('Bidirectional endpoint invocation completed successfully');
        return response;
    } catch (error: any) {
        console.error('Error invoking endpoint with bidirectional stream:', error);
        console.error('Error details:', {
            name: error.name,
            message: error.message,
            statusCode: error.$metadata?.httpStatusCode
        });
        throw error;
    }
}

// Main execution function
async function main(): Promise<void> {
    try {
        console.log('Starting SageMaker deployment process...');

        await invokeEndpointWithBidirectionalStream();

        console.log('All operations completed successfully!');

    } catch (error) {
        console.error('Deployment process failed:', error);
        throw error;
    }
}

declare const require: any;
declare const module: any;
declare const process: any;

if (typeof require !== 'undefined' && require.main === module) {
    main().catch(error => {
        console.error('Script execution failed:', error);
        if (typeof process !== 'undefined') {
            process.exit(1);
        }
    });
}

export {
    invokeEndpointWithBidirectionalStream,
    config,
    bidiEndpoint
};
