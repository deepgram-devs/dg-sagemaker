import {
    SageMakerRuntimeHTTP2Client,
    InvokeEndpointWithBidirectionalStreamCommand,
    type InvokeEndpointWithBidirectionalStreamCommandOutput,
    type RequestStreamEvent,
    type InvokeEndpointWithBidirectionalStreamCommandInput,
} from '@aws-sdk/client-sagemaker-runtime-http2';
import { playAudio } from '@mastra/node-audio';
import { Readable } from 'stream';
import { config, bidiEndpoint } from './tts';

// All English Deepgram aura-2 voices
const englishAura2Voices = [
    'aura-2-amalthea-en',
    'aura-2-andromeda-en',
    'aura-2-apollo-en',
    'aura-2-arcas-en',
    'aura-2-aries-en',
    'aura-2-asteria-en',
    'aura-2-athena-en',
    'aura-2-atlas-en',
    'aura-2-aurora-en',
    'aura-2-callista-en',
    'aura-2-cora-en',
    'aura-2-cordelia-en',
    'aura-2-delia-en',
    'aura-2-draco-en',
    'aura-2-electra-en',
    'aura-2-harmonia-en',
    'aura-2-helena-en',
    'aura-2-hera-en',
    'aura-2-hermes-en',
    'aura-2-hyperion-en',
    'aura-2-iris-en',
    'aura-2-janus-en',
    'aura-2-juno-en',
    'aura-2-jupiter-en',
    'aura-2-luna-en',
    'aura-2-mars-en',
    'aura-2-minerva-en',
    'aura-2-neptune-en',
    'aura-2-odysseus-en',
    'aura-2-ophelia-en',
    'aura-2-orion-en',
    'aura-2-orpheus-en',
    'aura-2-pandora-en',
    'aura-2-phoebe-en',
    'aura-2-pluto-en',
    'aura-2-saturn-en',
    'aura-2-selene-en',
    'aura-2-thalia-en',
    'aura-2-theia-en',
    'aura-2-vesta-en',
    'aura-2-zeus-en',
];

// Configuration from tts.ts
const region: string = "us-east-2";
const modelInvocationPath = 'v1/speak';
const inputText = `
The morning sun, filtered through the rustling leaves of the Sector 5 Slums. The beautiful trees swayed in the wind while Aeris watched them.
`;

const SageMakerRuntimeHTTP2ClientForBidi = new SageMakerRuntimeHTTP2Client({
    region: region,
    endpoint: bidiEndpoint,
    requestHandler: {
        requestTimeout: 30000,
    },
    // Enable logger to see request details (optional)
    // logger: console
});

// Enable request logging to see the actual URL being used
process.env.AWS_SDK_JS_SUPPRESS_MAINTENANCE_MODE_MESSAGE = 'true';

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms)).then(() => console.log("Sleeping completed!"));

async function* createTTSRequestStream() {
    const words = inputText.trim().split(/\s+/);
    const chunkSize = 20;
    const chunks: string[] = [];

    for (let i = 0; i < words.length; i += chunkSize) {
        const chunk = words.slice(i, i + chunkSize).join(' ');
        chunks.push(chunk);
    }

    console.log(`Splitting text into ${chunks.length} chunks of ~${chunkSize} words each`);

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

    await sleep(5000);
}

async function invokeEndpointWithVoice(voiceId: string): Promise<InvokeEndpointWithBidirectionalStreamCommandOutput> {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`Testing voice: ${voiceId}`);
    console.log('='.repeat(60));

    const modelQueryString = `model=${voiceId}`;

    const invokeParams: InvokeEndpointWithBidirectionalStreamCommandInput = {
        EndpointName: config.endpointName,
        Body: createTTSRequestStream(),
        ModelInvocationPath: modelInvocationPath,
        ModelQueryString: modelQueryString,
    };

    try {
        console.log('Endpoint name:', config.endpointName);
        console.log('Model invocation path:', modelInvocationPath);
        console.log('Model query string:', modelQueryString);
        console.log('Bidi endpoint:', bidiEndpoint);
        const command = new InvokeEndpointWithBidirectionalStreamCommand(invokeParams);
        console.log('Command params (endpoint, path, query):', {
            EndpointName: invokeParams.EndpointName,
            ModelInvocationPath: invokeParams.ModelInvocationPath,
            ModelQueryString: invokeParams.ModelQueryString
        });
        console.log('Sending bidirectional stream request...');
        console.log('NOTE: The SDK will construct the HTTP request URL internally.');
        console.log('Expected URL pattern: {bidiEndpoint}/endpoints/{endpointName}/invocations-bidirectional');
        const response: InvokeEndpointWithBidirectionalStreamCommandOutput = await SageMakerRuntimeHTTP2ClientForBidi.send(command);
        console.log('Response metadata:', JSON.stringify(response.$metadata, null, 2));
        if (response.$metadata?.httpStatusCode) {
            console.log('HTTP Status Code:', response.$metadata.httpStatusCode);
        }
        if (response.$metadata?.requestId) {
            console.log('Request ID:', response.$metadata.requestId);
        }

        console.log('Bidirectional stream response received. Processing...');

        if (response.Body) {
            let chunkCount = 0;
            const timeout = setTimeout(() => {
                console.log('Timeout waiting for bidirectional stream chunks');
            }, 20000);

            const audioChunks: Buffer[] = [];

            try {
                for await (const chunk of response.Body) {
                    chunkCount++;
                    console.log(`Processing bidirectional chunk ${chunkCount}:`, Object.keys(chunk));

                    if (chunk.PayloadPart && chunk.PayloadPart.Bytes) {
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

                if (audioChunks.length > 0) {
                    console.log(`Playing ${audioChunks.length} audio chunks for voice ${voiceId}...`);
                    const audioBuffer = Buffer.concat(audioChunks);

                    const audioStream = Readable.from(audioBuffer);

                    const speaker = playAudio(audioStream, {
                        channels: 1,
                        bitDepth: 16,
                        sampleRate: 24000,
                    });

                    await new Promise<void>((resolve, reject) => {
                        speaker.on('close', () => {
                            console.log(`Audio playback completed for ${voiceId}`);
                            resolve();
                        });
                        speaker.on('error', (err) => {
                            console.error(`Audio playback error for ${voiceId}:`, err);
                            reject(err);
                        });
                    });
                } else {
                    console.log(`No audio chunks received for voice ${voiceId}`);
                }
            } catch (streamError) {
                clearTimeout(timeout);
                console.error(`Error processing bidirectional stream for ${voiceId}:`, streamError);
                throw streamError;
            }
        } else {
            console.log('No bidirectional response body received');
        }

        console.log(`Voice ${voiceId} completed successfully`);
        return response;
    } catch (error: any) {
        console.error(`Error invoking endpoint with voice ${voiceId}:`, error);
        console.error('Error details:', {
            name: error.name,
            message: error.message,
            statusCode: error.$metadata?.httpStatusCode
        });
        throw error;
    }
}

async function testAllVoices(): Promise<void> {
    console.log(`Starting test of ${englishAura2Voices.length} Deepgram aura-2 English voices...\n`);

    const results: { voice: string; success: boolean; error?: string }[] = [];

    for (const voice of englishAura2Voices) {
        try {
            await invokeEndpointWithVoice(voice);
            results.push({ voice, success: true });
            console.log(`✓ ${voice} - Success\n`);
        } catch (error: any) {
            results.push({
                voice,
                success: false,
                error: error.message || 'Unknown error'
            });
            console.error(`✗ ${voice} - Failed: ${error.message}\n`);
        }

        // Add a small delay between voices to avoid overwhelming the endpoint
        await sleep(2000);
    }

    // Print summary
    console.log('\n' + '='.repeat(60));
    console.log('TEST SUMMARY');
    console.log('='.repeat(60));
    console.log(`Total voices tested: ${results.length}`);
    console.log(`Successful: ${results.filter(r => r.success).length}`);
    console.log(`Failed: ${results.filter(r => !r.success).length}`);

    const failures = results.filter(r => !r.success);
    if (failures.length > 0) {
        console.log('\nFailed voices:');
        failures.forEach(f => {
            console.log(`  - ${f.voice}: ${f.error}`);
        });
    }
}

// Main execution
async function main(): Promise<void> {
    try {
        await testAllVoices();
        console.log('\nAll voice tests completed!');
    } catch (error) {
        console.error('Voice testing process failed:', error);
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

export { testAllVoices, englishAura2Voices };
