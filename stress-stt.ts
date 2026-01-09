// Prerequisites: corepack enable && yarn && npm install -g tsx
// Run with: tsx parallel-stt.ts

import { invokeEndpointWithBidirectionalStream } from './stt.file';

// Number of parallel invocations
const PARALLEL_COUNT = 20;

// Main execution function
async function runParallelInvocations(): Promise<void> {
    console.log(`Starting ${PARALLEL_COUNT} parallel invocations...`);
    const startTime = Date.now();

    // Create an array of promises for parallel execution
    const promises = Array.from({ length: PARALLEL_COUNT }, (_, index) => {
        const invocationNumber = index + 1;
        console.log(`Launching invocation ${invocationNumber}/${PARALLEL_COUNT}`);

        return invokeEndpointWithBidirectionalStream()
            .then(() => {
                console.log(`✓ Invocation ${invocationNumber} completed successfully`);
                return { success: true, invocationNumber };
            })
            .catch((error) => {
                console.error(`✗ Invocation ${invocationNumber} failed:`, error.message);
                return { success: false, invocationNumber, error };
            });
    });

    // Wait for all promises to settle
    console.log(`\nWaiting for all ${PARALLEL_COUNT} invocations to complete...\n`);
    const results = await Promise.all(promises);

    // Calculate statistics
    const endTime = Date.now();
    const duration = (endTime - startTime) / 1000;
    const successCount = results.filter(r => r.success).length;
    const failureCount = results.filter(r => !r.success).length;

    // Print summary
    console.log('\n' + '='.repeat(60));
    console.log('PARALLEL INVOCATION SUMMARY');
    console.log('='.repeat(60));
    console.log(`Total invocations: ${PARALLEL_COUNT}`);
    console.log(`Successful: ${successCount}`);
    console.log(`Failed: ${failureCount}`);
    console.log(`Total duration: ${duration.toFixed(2)} seconds`);
    console.log(`Average time per invocation: ${(duration / PARALLEL_COUNT).toFixed(2)} seconds`);
    console.log('='.repeat(60));

    // List failed invocations if any
    if (failureCount > 0) {
        console.log('\nFailed invocations:');
        results
            .filter(r => !r.success)
            .forEach(r => {
                console.log(`  - Invocation ${r.invocationNumber}: ${r.error?.message || 'Unknown error'}`);
            });
    }

    // Exit with error code if any invocations failed
    if (failureCount > 0) {
        console.error(`\n${failureCount} invocation(s) failed`);
        if (typeof process !== 'undefined') {
            process.exit(1);
        }
    } else {
        console.log('\nAll invocations completed successfully!');
    }
}

// Run the script if this file is executed directly
declare const require: any;
declare const module: any;
declare const process: any;

if (typeof require !== 'undefined' && require.main === module) {
    runParallelInvocations().catch(error => {
        console.error('Script execution failed:', error);
        if (typeof process !== 'undefined') {
            process.exit(1);
        }
    });
}

export { runParallelInvocations };
