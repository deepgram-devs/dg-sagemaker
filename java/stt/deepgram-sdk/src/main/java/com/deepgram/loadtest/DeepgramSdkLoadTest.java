package com.deepgram.loadtest;

import com.deepgram.resources.listen.v1.websocket.V1ConnectOptions;
import com.deepgram.types.ListenV1Channels;
import com.deepgram.types.ListenV1Encoding;
import com.deepgram.types.ListenV1InterimResults;
import com.deepgram.types.ListenV1Model;
import com.deepgram.types.ListenV1SampleRate;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

/**
 * Deepgram SDK SageMaker Load Test
 *
 * Uses the Deepgram Java SDK plus the Deepgram SageMaker transport
 * (https://github.com/deepgram/deepgram-java-sdk-transport-sagemaker)
 * to validate high-concurrency bidi streaming against a SageMaker endpoint.
 *
 * SDK-level equivalent of the raw AWS SDK load test under ../aws-sdk.
 * Exercises the full path: DeepgramClient -> SageMakerTransportFactory ->
 * SageMakerTransport -> SageMakerRuntimeHttp2AsyncClient.
 */
@Command(name = "dg-sdk-loadtest",
    description = "Deepgram SDK SageMaker Load Test -- bidi streaming via Deepgram Java SDK",
    mixinStandardHelpOptions = true,
    version = "1.0.0")
public class DeepgramSdkLoadTest implements Runnable {

    private static final Logger log = LoggerFactory.getLogger(DeepgramSdkLoadTest.class);

    @Parameters(index = "0", description = "SageMaker endpoint name")
    private String endpointName;

    @Option(names = {"--file", "-f"}, required = true, description = "Path to 16-bit PCM WAV file")
    private Path wavFile;

    @Option(names = {"--connections", "-c"}, defaultValue = "1",
        description = "Total simultaneous streaming connections (default: ${DEFAULT-VALUE})")
    private int connections;

    @Option(names = {"--batch-size"}, defaultValue = "0",
        description = "Connections to open per batch (0 = all at once, default: ${DEFAULT-VALUE})")
    private int batchSize;

    @Option(names = {"--batch-delay"}, defaultValue = "0",
        description = "Seconds to wait between batches (default: ${DEFAULT-VALUE})")
    private double batchDelay;

    @Option(names = {"--loop"}, defaultValue = "true", negatable = true,
        description = "Loop audio file continuously. Use --no-loop to play once (default: ${DEFAULT-VALUE})")
    private boolean loop;

    @Option(names = {"--duration"}, defaultValue = "0",
        description = "Stop after N seconds (0 = run until audio ends or Ctrl+C, default: ${DEFAULT-VALUE})")
    private int durationSeconds;

    @Option(names = {"--region"}, defaultValue = "us-west-2",
        description = "AWS region (default: ${DEFAULT-VALUE})")
    private String region;

    @Option(names = {"--max-retries"}, defaultValue = "10",
        description = "Max retries per connection on retryable errors (default: ${DEFAULT-VALUE})")
    private int maxRetries;

    @Option(names = {"--await-final-results"}, defaultValue = "15",
        description = "After sendCloseStream, wait up to N seconds for the model to flush "
                    + "remaining transcripts and close the WebSocket. Returns sooner if the "
                    + "server closes early. Bump this when retries cause large model backlogs "
                    + "and the default 15 s clips tail-end transcripts (default: ${DEFAULT-VALUE})")
    private int awaitFinalResultsSeconds;

    // --- Deepgram model params ---
    @Option(names = {"--model"}, defaultValue = "nova-3",
        description = "Deepgram model: nova-3, nova-2, etc. (default: ${DEFAULT-VALUE})")
    private String model;

    @Option(names = {"--interim-results"}, defaultValue = "false",
        description = "Enable interim/partial results (default: ${DEFAULT-VALUE})")
    private boolean interimResults;

    @Option(names = {"--log-level"}, defaultValue = "INFO",
        description = "Log level: DEBUG, INFO, WARN, ERROR (default: ${DEFAULT-VALUE})")
    private String logLevel;

    private final AtomicBoolean running = new AtomicBoolean(true);

    public static void main(String[] args) {
        int exitCode = new CommandLine(new DeepgramSdkLoadTest()).execute(args);
        System.exit(exitCode);
    }

    @Override
    public void run() {
        configureLogLevel();
        validateInputs();

        int effectiveBatchSize = batchSize > 0 ? batchSize : connections;

        System.err.println("=== Deepgram SDK SageMaker Load Test ===");
        System.err.println("Endpoint:       " + endpointName);
        System.err.println("WAV file:       " + wavFile);
        System.err.println("Connections:    " + connections);
        System.err.println("Batch size:     " + effectiveBatchSize);
        System.err.println("Batch delay:    " + batchDelay + "s");
        System.err.println("Region:         " + region);
        System.err.println("Model:          " + model);
        System.err.println("Loop:           " + loop);
        System.err.println("Max retries:    " + maxRetries);
        System.err.println("SDK:            deepgram-sagemaker-transport (local build with maxStreams=1 fix)");
        System.err.println();

        printWavInfo();

        // Read WAV format for connect options
        int sampleRate = 16000;
        int channels = 1;
        try (AudioInputStream ais = AudioSystem.getAudioInputStream(wavFile.toFile())) {
            AudioFormat fmt = ais.getFormat();
            sampleRate = (int) fmt.getSampleRate();
            channels = fmt.getChannels();
        } catch (Exception e) {
            System.err.println("WARNING: Could not read WAV format, using defaults");
        }

        V1ConnectOptions connectOptions = V1ConnectOptions.builder()
                .model(ListenV1Model.valueOf(model))
                .encoding(ListenV1Encoding.LINEAR16)
                .sampleRate(ListenV1SampleRate.of(sampleRate))
                .channels(ListenV1Channels.of(channels))
                .interimResults(interimResults ? ListenV1InterimResults.TRUE : ListenV1InterimResults.FALSE)
                .build();

        Runtime.getRuntime().addShutdownHook(new Thread(() -> running.set(false)));

        List<SdkStreamingConnection> allConnections = new ArrayList<>();
        List<CompletableFuture<Void>> futures = new ArrayList<>();

        // Default loop-stop policy: when --duration is not set and --loop is on,
        // stop all connections once every connection has completed one full read
        // of the WAV file. This bounds the run without Ctrl+C.
        java.util.concurrent.atomic.AtomicInteger firstPassCount = new java.util.concurrent.atomic.AtomicInteger();
        boolean stopOnAllFirstPasses = loop && durationSeconds == 0;
        Runnable onFirstPass = () -> {
            int done = firstPassCount.incrementAndGet();
            if (stopOnAllFirstPasses && done >= connections) {
                System.err.println();
                System.err.println("All " + connections + " connection(s) completed one full pass. Stopping...");
                running.set(false);
                allConnections.forEach(SdkStreamingConnection::stop);
            }
        };

        long testStartNanos = System.nanoTime();

        try {
            int numBatches = (connections + effectiveBatchSize - 1) / effectiveBatchSize;
            System.err.println("Opening " + connections + " connection(s) in " +
                numBatches + " batch(es) (one SageMaker client per connection)...");

            ScheduledExecutorService dashboard = Executors.newSingleThreadScheduledExecutor(r -> {
                Thread t = new Thread(r, "dashboard");
                t.setDaemon(true);
                return t;
            });
            dashboard.scheduleAtFixedRate(
                () -> printStatus(allConnections, testStartNanos),
                2, 2, TimeUnit.SECONDS
            );

            if (durationSeconds > 0) {
                dashboard.schedule(() -> {
                    System.err.println("\nDuration limit reached (" + durationSeconds + "s). Stopping...");
                    running.set(false);
                    allConnections.forEach(SdkStreamingConnection::stop);
                }, durationSeconds, TimeUnit.SECONDS);
            }

            for (int batchStart = 0; batchStart < connections; batchStart += effectiveBatchSize) {
                if (!running.get()) break;
                int batchEnd = Math.min(batchStart + effectiveBatchSize, connections);
                int batchNum = batchStart / effectiveBatchSize + 1;

                System.err.printf("Opening batch %d/%d: connections %d-%d...%n",
                    batchNum, numBatches, batchStart + 1, batchEnd);

                for (int i = batchStart; i < batchEnd; i++) {
                    SdkStreamingConnection conn = new SdkStreamingConnection(
                        i + 1, endpointName, region, wavFile,
                        connectOptions, loop, maxRetries, awaitFinalResultsSeconds, onFirstPass
                    );
                    allConnections.add(conn);
                    futures.add(conn.start());
                }

                if (batchEnd < connections && batchDelay > 0) {
                    Thread.sleep((long) (batchDelay * 1000));
                }
            }

            System.err.println("All " + connections + " connection(s) launched. Streaming...");
            System.err.println();

            CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
                .exceptionally(t -> null)
                .join();

            dashboard.shutdownNow();

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Interrupted.");
        } finally {
            allConnections.forEach(SdkStreamingConnection::stop);
        }

        double wallTime = (System.nanoTime() - testStartNanos) / 1_000_000_000.0;
        printSummary(allConnections, wallTime);
    }

    private void printWavInfo() {
        try (AudioInputStream ais = AudioSystem.getAudioInputStream(wavFile.toFile())) {
            AudioFormat fmt = ais.getFormat();
            long frames = ais.getFrameLength();
            double duration = frames / fmt.getSampleRate();
            System.err.printf("WAV: %s | %.0f Hz | %dch | %.2fs%n",
                wavFile.getFileName(), fmt.getSampleRate(), fmt.getChannels(), duration);
        } catch (Exception e) {
            System.err.println("WAV: " + wavFile + " (could not read format)");
        }
    }

    private void printStatus(List<SdkStreamingConnection> conns, long testStartNanos) {
        if (conns.isEmpty()) return;
        long activeCount = conns.stream().filter(SdkStreamingConnection::isActive).count();
        long errorCount = conns.stream().filter(SdkStreamingConnection::isErrored).count();
        int totalTranscripts = conns.stream().mapToInt(SdkStreamingConnection::getTranscriptCount).sum();
        int totalChunks = conns.stream().mapToInt(SdkStreamingConnection::getChunkCount).sum();
        int totalRetries = conns.stream().mapToInt(SdkStreamingConnection::getRetryCount).sum();
        double elapsed = (System.nanoTime() - testStartNanos) / 1_000_000_000.0;

        System.err.printf("\r[%s] Active: %d/%d | Errored: %d | Retries: %d | Transcripts: %d | Chunks: %d    ",
            formatDuration(elapsed), activeCount, conns.size(), errorCount, totalRetries, totalTranscripts, totalChunks);
    }

    private void printSummary(List<SdkStreamingConnection> conns, double wallTime) {
        System.err.println();
        System.err.println();
        System.err.println("=== STREAM SUMMARY ===");

        long successful = conns.stream().filter(c -> !c.isErrored()).count();
        long errored = conns.stream().filter(SdkStreamingConnection::isErrored).count();
        int totalTranscripts = conns.stream().mapToInt(SdkStreamingConnection::getTranscriptCount).sum();
        int totalChunks = conns.stream().mapToInt(SdkStreamingConnection::getChunkCount).sum();
        int totalRetries = conns.stream().mapToInt(SdkStreamingConnection::getRetryCount).sum();

        System.err.println("Total connections:  " + conns.size());
        System.err.println("Successful:         " + successful);
        System.err.println("Errored:            " + errored);
        System.err.println("Total retries:      " + totalRetries);
        System.err.println("Total transcripts:  " + totalTranscripts);
        System.err.println("Total chunks sent:  " + totalChunks);
        System.err.printf("Wall time:          %.2fs%n", wallTime);
        System.err.println();

        List<Double> durations = conns.stream()
            .filter(c -> !c.isErrored())
            .map(SdkStreamingConnection::getDurationSeconds)
            .sorted()
            .collect(Collectors.toList());

        if (!durations.isEmpty()) {
            System.err.println("--- Connection Durations (successful) ---");
            System.err.printf("Min:    %.2fs%n", durations.get(0));
            System.err.printf("Median: %.2fs%n", durations.get(durations.size() / 2));
            System.err.printf("Max:    %.2fs%n", durations.get(durations.size() - 1));
            System.err.printf("Mean:   %.2fs%n", durations.stream().mapToDouble(d -> d).average().orElse(0));
        }

        if (errored > 0) {
            System.err.println();
            System.err.println("--- Errors ---");
            Map<String, Long> errorCounts = conns.stream()
                .filter(SdkStreamingConnection::isErrored)
                .flatMap(c -> c.getErrorMessages().stream())
                .collect(Collectors.groupingBy(m -> truncate(m, 120), Collectors.counting()));
            errorCounts.entrySet().stream()
                .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
                .forEach(e -> System.err.printf("  [%dx] %s%n", e.getValue(), e.getKey()));
        }
    }

    private void validateInputs() {
        if (!Files.exists(wavFile)) {
            System.err.println("ERROR: WAV file not found: " + wavFile);
            System.exit(1);
        }
        if (connections < 1) {
            System.err.println("ERROR: --connections must be >= 1");
            System.exit(1);
        }
    }

    private void configureLogLevel() {
        ch.qos.logback.classic.Logger appLogger =
            (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("com.deepgram.loadtest");
        switch (logLevel.toUpperCase()) {
            case "DEBUG": appLogger.setLevel(ch.qos.logback.classic.Level.DEBUG); break;
            case "WARN":  appLogger.setLevel(ch.qos.logback.classic.Level.WARN); break;
            case "ERROR": appLogger.setLevel(ch.qos.logback.classic.Level.ERROR); break;
            default:      appLogger.setLevel(ch.qos.logback.classic.Level.INFO); break;
        }
    }

    private static String formatDuration(double seconds) {
        int h = (int) (seconds / 3600);
        int m = (int) ((seconds % 3600) / 60);
        int s = (int) (seconds % 60);
        return String.format("%02d:%02d:%02d", h, m, s);
    }

    private static String truncate(String s, int maxLen) {
        return s.length() <= maxLen ? s : s.substring(0, maxLen) + "...";
    }
}
