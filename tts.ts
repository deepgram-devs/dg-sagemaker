import {
    SageMakerRuntimeHTTP2Client,
    InvokeEndpointWithBidirectionalStreamCommand,
    type InvokeEndpointWithBidirectionalStreamCommandOutput,
    type RequestStreamEvent,
    type InvokeEndpointWithBidirectionalStreamCommandInput,
} from '@aws-sdk/client-sagemaker-runtime-http2';
import { playAudio } from '@mastra/node-audio';
import { Readable } from 'stream';

// Configuration interface
interface Config {
    endpointName: string;
}

// Configuration
const region: string = "us-east-2"; // CHANGE ME: Specify the region where your SageMake Endpoint is deployed.
const bidiEndpoint: string = `https://runtime.sagemaker.${region}.amazonaws.com:8443`; // CHANGE ME: This must correspond to the AWS region where your Endpoint is deployed.
const modelInvocationPath = 'v1/speak'; // The internal WebSocket API route you want to access, used by Deepgram specifically.
// const modelInvocationPath = '/invocations-bidirectional'; // The internal WebSocket API route you want to access, used by Deepgram specifically.
const modelQueryString = 'model=aura-2-thalia-en'; // CHANGE ME: Update this to the model parameters you want. Preview only supports nova-3, entity detection, and diarization.

const inputText = `
The morning sun, filtered through the rustling leaves of the Sector 5 Slums. The beautiful trees swayed in the wind while Aeris watched them.
`;


const config: Config = {
    endpointName: `naomi-tts-real`, // CHANGE ME: Update this value to the name of the SageMaker Endpoint you deploy
};

const SageMakerRuntimeHTTP2ClientForBidi = new SageMakerRuntimeHTTP2Client({
    region: region,
    endpoint: bidiEndpoint
});

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms)).then(() => console.log("Sleeping completed!"));

async function* createTTSRequestStream() {
    // Split the input text into chunks of 5 words each
    const words = inputText.trim().split(/\s+/);
    const chunkSize = 20;
    const chunks: string[] = [];
    
    for (let i = 0; i < words.length; i += chunkSize) {
        const chunk = words.slice(i, i + chunkSize).join(' ');
        chunks.push(chunk);
    }
    
    console.log(`Splitting text into ${chunks.length} chunks of ~${chunkSize} words each`);
    
    // Yield each chunk as a separate PayloadPart
    for (let i = 0; i < chunks.length; i++) {
        const isLastChunk = i === chunks.length - 1;
        const ttsPayload = {
            type: "Speak",
            text: chunks[i]
        };

        if (chunks.length > i && chunks[i]) {
            console.log(`Sending chunk ${i + 1}/${chunks.length}: "${chunks[i]!.substring(0, 50)}..."`);
        }
        
        yield {
            PayloadPart: {
                Bytes: new TextEncoder().encode(JSON.stringify(ttsPayload)),
                DataType: "UTF8",
            },
        };
        await sleep(3500);
        console.log('Sending flush message');
        yield {
            PayloadPart: {
                Bytes: new TextEncoder().encode(JSON.stringify({"type": "Flush"})),
                DataType: "UTF8",
            },            

    }
    }
    
    // Sleep to ensure the complete response stream is received
    // The output stream will be closed as soon as input stream terminates,
    // so we need to wait long enough for the TTS audio to be generated and streamed back
    await sleep(10000); // 10 seconds should be enough for the TTS to complete
}

async function invokeEndpointWithBidirectionalStream(): Promise<InvokeEndpointWithBidirectionalStreamCommandOutput> {
    console.log('Invoking endpoint with bidirectional stream...');

    const invokeParams: InvokeEndpointWithBidirectionalStreamCommandInput = {
        EndpointName: config.endpointName,
        Body: createTTSRequestStream(), // as AsyncIterable<RequestStreamEvent>
        ModelInvocationPath: modelInvocationPath,
        ModelQueryString: modelQueryString,
    };

    try {
        console.log('Using custom bidi endpoint:', bidiEndpoint);
        const command = new InvokeEndpointWithBidirectionalStreamCommand(invokeParams);
        console.log('Sending bidirectional stream request...');
        const response: InvokeEndpointWithBidirectionalStreamCommandOutput = await SageMakerRuntimeHTTP2ClientForBidi.send(command);

        console.log('Bidirectional stream response received. Processing...');
        console.log('Response metadata:', response.$metadata);

        if (response.Body) {
            let chunkCount = 0;
            const timeout = setTimeout(() => {
                console.log('Timeout waiting for bidirectional stream chunks');
            }, 20000); // 20 second timeout

            // Create a readable stream to collect audio chunks
            const audioChunks: Buffer[] = [];
            
            try {
                // Read responses from the bidirectional stream
                for await (const chunk of response.Body) {
                    chunkCount++;
                    console.log(`Processing bidirectional chunk ${chunkCount}:`, Object.keys(chunk));

                    if (chunk.PayloadPart && chunk.PayloadPart.Bytes) {
                        // Check if this is audio data (binary) or metadata (text)
                            // If decoding fails, it's likely binary audio data
                        console.log(`Received audio chunk ${chunkCount}, size: ${chunk.PayloadPart.Bytes.length} bytes`);
                        if (chunk.PayloadPart.Bytes.length != 34) {
                            audioChunks.push(Buffer.from(chunk.PayloadPart.Bytes));
                        }
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

                // Play the collected audio if we have any
                if (audioChunks.length > 0) {
                    console.log(`Playing ${audioChunks.length} audio chunks...`);
                    const audioBuffer = Buffer.concat(audioChunks);
                    
                    // Create a readable stream from the audio buffer
                    const audioStream = Readable.from(audioBuffer);
                    
                    // Play the audio through the default audio device
                    // Deepgram typically returns audio in linear16 format at 24000 Hz
                    const speaker = playAudio(audioStream, {
                        channels: 1,          // Mono audio
                        bitDepth: 16,         // 16-bit audio
                        sampleRate: 24000,    // 24kHz sample rate (Deepgram default)
                    });

                    // Wait for audio playback to complete
                    await new Promise<void>((resolve, reject) => {
                        speaker.on('close', () => {
                            console.log('Audio playback completed');
                            resolve();
                        });
                        speaker.on('error', (err) => {
                            console.error('Audio playback error:', err);
                            reject(err);
                        });
                    });
                } else {
                    console.log('No audio chunks received to play');
                }
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

console.log(`Require is: ${typeof require}`);

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
