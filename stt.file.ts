
// Prerequisites: corepack enable && yarn && npm install -g tsx

import {
    SageMakerRuntimeHTTP2Client,
    InvokeEndpointWithBidirectionalStreamCommand,
    InvokeEndpointWithBidirectionalStreamCommandOutput,
    RequestStreamEvent,
    InvokeEndpointWithBidirectionalStreamCommandInput
} from '@aws-sdk/client-sagemaker-runtime-http2';
  import * as fs from 'fs';
import * as path from 'path';

// Configuration interface
interface Config {
    endpointName: string;
}

// Configuration
const region: string = "us-west-2"; // CHANGE ME: Specify the region where your SageMake Endpoint is deployed.
const bidiEndpoint: string = `https://runtime.sagemaker.${region}.amazonaws.com:8443`; // CHANGE ME: This must correspond to the AWS region where your Endpoint is deployed.
const inputFilePath = `/Users/trev/2025-12-15-aeris-story.wav`; // CHANGE ME: The local filesystem path to the audio file you want to transcribe.
const modelInvocationPath = 'v1/listen'; // The internal WebSocket API route you want to access, used by Deepgram specifically.
const modelQueryString = 'model=nova-3&diarize=true&language=multi'; // CHANGE ME: Update this to the model parameters you want. Preview only supports nova-3, entity detection, and diarization.

const config: Config = {
    endpointName: `2026-01-07-trevor-auto-scaling-test`, // CHANGE ME: Update this value to the name of the SageMaker Endpoint you deploy
};

const sagemakerRuntimeClientForBidi = new SageMakerRuntimeHTTP2Client({
    region: region,
    endpoint: bidiEndpoint
});

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms)).then(() => console.log("Sleeping completed!"));

// Create an async iterable for streaming WAV file chunks from the local filesystem
async function* createWavFileRequestStream(wavFilePath: string, chunkSize: number = 1024*1024) { // Default 1MB chunks
    const KEEPALIVE_INTERVAL = 3000; // 3 seconds
    let lastKeepaliveTime = Date.now();
    
    // Resolve the full path to the WAV file
    const fullPath = path.resolve(wavFilePath);
    
    // Check if file exists
    if (!fs.existsSync(fullPath)) {
        throw new Error(`WAV file not found: ${fullPath}`);
    }
    
    console.log(`Starting to stream WAV file: ${fullPath}`);
    const fileStats = fs.statSync(fullPath);
    console.log(`File size: ${fileStats.size} bytes`);
    
    // Create a read stream for the WAV file
    const readStream = fs.createReadStream(fullPath, { highWaterMark: chunkSize });
    
    let chunkNumber = 0;
    let totalBytesRead = 0;
    let streamActive = true;
    
    // Set up keepalive interval
    const keepaliveInterval = setInterval(() => {
        if (streamActive && Date.now() - lastKeepaliveTime >= KEEPALIVE_INTERVAL) {
            console.log('Sending keepalive message...');
            lastKeepaliveTime = Date.now();
        }
    }, KEEPALIVE_INTERVAL);
    
    try {
        for await (const chunk of readStream) {
            chunkNumber++;
            totalBytesRead += chunk.length;
            
            // Check if we need to send a keepalive
            const now = Date.now();
            if (now - lastKeepaliveTime >= KEEPALIVE_INTERVAL) {
                const timestamp = new Date(now).toISOString();
                console.log(`Sending keepalive message at ${timestamp}...`);
                yield {
                    PayloadPart: {
                        Bytes: new TextEncoder().encode(JSON.stringify({
                            type: "KeepAlive",
                            timestamp: timestamp
                        })),
                        DataType: "UTF8",
                    },
                };
                lastKeepaliveTime = now;
            }
            
            // Determine completion state
            const isLastChunk = totalBytesRead >= fileStats.size;
            const completionState = isLastChunk ? "COMPLETE" : "PARTIAL";
            
            console.log(`Streaming WAV chunk ${chunkNumber}: ${chunk.length} bytes (${totalBytesRead}/${fileStats.size})`);
            
            yield {
                PayloadPart: {
                    Bytes: chunk,
                    DataType: "BINARY",
                    // CompletionState: completionState,
                },
            };
            
            lastKeepaliveTime = Date.now();
        }
        
        console.log(`Finished streaming WAV file: ${chunkNumber} chunks, ${totalBytesRead} total bytes`);
        
        // Continue sending keepalive messages after audio streaming is complete
        console.log('Audio streaming complete. Continuing to send keepalive messages...');
        const keepaliveEndTime = Date.now() + 120000; // Keep alive for 120 seconds
        
        while (Date.now() < keepaliveEndTime) {
            const now = Date.now();
            if (now - lastKeepaliveTime >= KEEPALIVE_INTERVAL) {
                const timestamp = new Date(now).toISOString();
                console.log(`Sending post-stream keepalive message at ${timestamp}...`);
                yield {
                    PayloadPart: {
                        Bytes: new TextEncoder().encode(JSON.stringify({
                            type: "KeepAlive",
                            timestamp: timestamp
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

async function invokeEndpointWithBidirectionalStream(): Promise<InvokeEndpointWithBidirectionalStreamCommandOutput> {
    console.log('Invoking endpoint with bidirectional stream...');

    // Create an async iterable for the request stream
    async function* createRequestStream() {
        yield {
            PayloadPart: {
                Bytes: new TextEncoder().encode('{"part": "1"}'),
                DataType: "UTF8",
            },
        };
        yield {
            PayloadPart: {
                Bytes: new TextEncoder().encode('{"part": "2"}'),
                DataType: "BINARY",
            },
        };
        yield {
            PayloadPart: {
                Bytes: new TextEncoder().encode('{"part": "3"}'),
                DataType: "UTF8",
                CompletionState: "PARTIAL",
            },
        };
        yield {
            PayloadPart: {
                Bytes: new TextEncoder().encode('{"part": "4"}'),
                DataType: "UTF8",
                CompletionState: "COMPLETE",
            },
        };
        yield {
            PayloadPart: {
                Bytes: new TextEncoder().encode('{"part": "5"}'),
                DataType: "BINARY",
                CompletionState: "PARTIAL",
            },
        };
        yield {
            PayloadPart: {
                Bytes: new TextEncoder().encode('{"part": "6"}'),
                DataType: "BINARY",
                CompletionState: "COMPLETE",
            },
        };
        // To prevent resource leak, the output stream will be closed as soon as input stream terminates.
        // 
        // In production, this will not cause problems, because both the input stream and output stream are
        // long-lived: thinking about the scenario where a voice-agent collects series of input binary audio
        // chunks from a microphone as a long-lived stream.
        //
        // However, in this demo, without sleeping, the input stream will be closed immediately after the
        // 6 payload parts are sent, hence the output stream will also be closed immediately, possibly before 
        // the model container starts to stream any response back. If this happens, we will not receive any
        // reponse payload parts. 
        // 
        // So, we need to sleep for long enough time (2 sec) to ensure the complete reponse stream is received.
        await sleep(2000);
    }

    const invokeParams: InvokeEndpointWithBidirectionalStreamCommandInput = {
        EndpointName: config.endpointName,
        // Body: createRequestStream() // as AsyncIterable<RequestEventStream>
        Body: createWavFileRequestStream(inputFilePath), // as AsyncIterable<RequestEventStream>
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

// Run the script if this file is executed directly
// Note: This check works in Node.js environments
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