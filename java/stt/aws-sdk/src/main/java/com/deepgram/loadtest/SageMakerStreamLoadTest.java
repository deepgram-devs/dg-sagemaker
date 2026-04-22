package com.deepgram.loadtest;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import software.amazon.awssdk.http.Protocol;
import software.amazon.awssdk.http.nio.netty.Http2Configuration;
import software.amazon.awssdk.http.nio.netty.NettyNioAsyncHttpClient;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.sagemakerruntimehttp2.SageMakerRuntimeHttp2AsyncClient;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

/**
 * Deepgram SageMaker Java Load Test
 *
 * Streams a WAV file to a Deepgram model on SageMaker using the AWS SDK v2
 * HTTP/2 bidirectional streaming API. Designed to verify that the Java SDK
 * can sustain high concurrency (400+ streams) when configured with
 * maxStreams=1 (one bidi stream per TCP connection).
 *
 * This is the Java equivalent of dg-sagemaker/python-stt/stt_wav_stress.py.
 *
 * Key configuration:
 *  --max-streams-per-connection  Controls Http2Configuration.maxStreams().
 *      Default: 1 (one stream per TCP connection). This is the critical
 *      setting that prevents HTTP/2 multiplexing starvation.
 *  --max-concurrency  Controls NettyNioAsyncHttpClient.maxConcurrency().
 *      Must be >= --connections. Default: connections + 50 headroom.
 */
@Command(name = "dg-sagemaker-loadtest",
    description = "Deepgram SageMaker Java Load Test — bidi streaming via AWS SDK v2",
    mixinStandardHelpOptions = true,
    version = "1.0.0")
public class SageMakerStreamLoadTest implements Runnable {

    private static final Logger log = LoggerFactory.getLogger(SageMakerStreamLoadTest.class);

    // --- CLI options ---
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

    // --- HTTP/2 tuning (the key knobs) ---
    @Option(names = {"--max-streams-per-connection"}, defaultValue = "1",
        description = "HTTP/2 max concurrent streams per TCP connection (default: ${DEFAULT-VALUE}). " +
            "Set to 1 to force one bidi stream per connection — prevents multiplexing starvation.")
    private int maxStreamsPerConnection;

    @Option(names = {"--max-concurrency"}, defaultValue = "0",
        description = "Netty max concurrency (total in-flight requests). 0 = connections + 50. " +
            "Must be >= connections when maxStreamsPerConnection=1.")
    private int maxConcurrency;

    @Option(names = {"--connection-timeout"}, defaultValue = "30",
        description = "TCP connection timeout in seconds (default: ${DEFAULT-VALUE})")
    private int connectionTimeoutSec;

    @Option(names = {"--max-retries"}, defaultValue = "10",
        description = "Max retries per connection on retryable errors like ThrottlingException (default: ${DEFAULT-VALUE})")
    private int maxRetries;

    // --- Deepgram model params ---
    @Option(names = {"--model"}, defaultValue = "nova-3",
        description = "Deepgram model (default: ${DEFAULT-VALUE})")
    private String model;

    @Option(names = {"--language"}, defaultValue = "en",
        description = "Language code (default: ${DEFAULT-VALUE})")
    private String language;

    @Option(names = {"--punctuate"}, defaultValue = "true",
        description = "Enable punctuation (default: ${DEFAULT-VALUE})")
    private boolean punctuate;

    @Option(names = {"--diarize"}, defaultValue = "false",
        description = "Enable speaker diarization (default: ${DEFAULT-VALUE})")
    private boolean diarize;

    @Option(names = {"--interim-results"}, defaultValue = "false",
        description = "Enable interim/partial results (default: ${DEFAULT-VALUE})")
    private boolean interimResults;

    @Option(names = {"--smart-format"}, defaultValue = "false",
        description = "Enable smart formatting (default: ${DEFAULT-VALUE})")
    private boolean smartFormat;

    @Option(names = {"--keyterms"}, description = "Comma-separated keyterms to boost (nova-3)")
    private String keyterms;

    @Option(names = {"--log-level"}, defaultValue = "INFO",
        description = "Log level: DEBUG, INFO, WARN, ERROR (default: ${DEFAULT-VALUE})")
    private String logLevel;

    // --- State ---
    private final AtomicBoolean running = new AtomicBoolean(true);

    public static void main(String[] args) {
        int exitCode = new CommandLine(new SageMakerStreamLoadTest()).execute(args);
        System.exit(exitCode);
    }

    @Override
    public void run() {
        configureLogLevel();
        validateInputs();

        int effectiveMaxConcurrency = maxConcurrency > 0 ? maxConcurrency : connections + 50;
        int effectiveBatchSize = batchSize > 0 ? batchSize : connections;

        System.err.println("=== Deepgram SageMaker Java Load Test ===");
        System.err.println("Endpoint:       " + endpointName);
        System.err.println("WAV file:       " + wavFile);
        System.err.println("Connections:    " + connections);
        System.err.println("Batch size:     " + effectiveBatchSize);
        System.err.println("Batch delay:    " + batchDelay + "s");
        System.err.println("Region:         " + region);
        System.err.println("Model:          " + model);
        System.err.println("Loop:           " + loop);
        System.err.println();
        System.err.println("--- HTTP/2 Configuration ---");
        System.err.println("maxStreamsPerConnection: " + maxStreamsPerConnection);
        System.err.println("maxConcurrency:         " + effectiveMaxConcurrency);
        System.err.println("connectionTimeout:      " + connectionTimeoutSec + "s");
        System.err.println();

        printWavInfo();

        String queryString = buildQueryString();
        System.err.println("Query string:   " + queryString);
        System.err.println();

        // Handle Ctrl+C
        Runtime.getRuntime().addShutdownHook(new Thread(() -> running.set(false)));

        // One client per connection — each client gets its own Netty connection pool.
        // With maxStreams=1, each client opens exactly one TCP connection for its
        // single bidi stream. This mirrors the Python stress test pattern where
        // each connection gets its own CRT transport.
        List<SageMakerRuntimeHttp2AsyncClient> clients = new ArrayList<>();
        List<StreamingConnection> allConnections = new ArrayList<>();
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
                allConnections.forEach(StreamingConnection::stop);
            }
        };

        long testStartNanos = System.nanoTime();

        try {
            // Open connections in batches
            int numBatches = (connections + effectiveBatchSize - 1) / effectiveBatchSize;
            System.err.println("Opening " + connections + " connection(s) in " +
                numBatches + " batch(es) (one client per connection)...");

            // Start status dashboard
            ScheduledExecutorService dashboard = Executors.newSingleThreadScheduledExecutor(r -> {
                Thread t = new Thread(r, "dashboard");
                t.setDaemon(true);
                return t;
            });
            dashboard.scheduleAtFixedRate(
                () -> printStatus(allConnections, testStartNanos),
                2, 2, TimeUnit.SECONDS
            );

            // Duration-based auto-stop
            if (durationSeconds > 0) {
                dashboard.schedule(() -> {
                    System.err.println("\nDuration limit reached (" + durationSeconds + "s). Stopping...");
                    running.set(false);
                    allConnections.forEach(StreamingConnection::stop);
                }, durationSeconds, TimeUnit.SECONDS);
            }

            for (int batchStart = 0; batchStart < connections; batchStart += effectiveBatchSize) {
                if (!running.get()) break;
                int batchEnd = Math.min(batchStart + effectiveBatchSize, connections);
                int batchNum = batchStart / effectiveBatchSize + 1;

                System.err.printf("Opening batch %d/%d: connections %d-%d...%n",
                    batchNum, numBatches, batchStart + 1, batchEnd);

                for (int i = batchStart; i < batchEnd; i++) {
                    SageMakerRuntimeHttp2AsyncClient client = createClient(effectiveMaxConcurrency);
                    clients.add(client);

                    StreamingConnection conn = new StreamingConnection(
                        i + 1, client, endpointName, wavFile, queryString, loop, maxRetries,
                        onFirstPass
                    );
                    allConnections.add(conn);

                    CompletableFuture<Void> future = conn.start();
                    futures.add(future);
                }

                // Delay between batches
                if (batchEnd < connections && batchDelay > 0) {
                    Thread.sleep((long) (batchDelay * 1000));
                }
            }

            System.err.println("All " + connections + " connection(s) launched. Streaming...");
            System.err.println();

            // Wait for all to complete
            CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
                .exceptionally(t -> null)  // don't throw if some fail
                .join();

            dashboard.shutdownNow();

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Interrupted.");
        } finally {
            // Clean up
            allConnections.forEach(StreamingConnection::stop);
            clients.forEach(SageMakerRuntimeHttp2AsyncClient::close);
        }

        double wallTime = (System.nanoTime() - testStartNanos) / 1_000_000_000.0;
        printSummary(allConnections, wallTime);
    }

    /**
     * Create a SageMaker HTTP/2 async client with explicit Http2Configuration.
     *
     * The critical setting is maxStreams — controls how many HTTP/2 streams
     * share each TCP connection. With maxStreams=1, each bidi stream gets
     * its own TCP connection, preventing multiplexing starvation.
     */
    private SageMakerRuntimeHttp2AsyncClient createClient(int effectiveMaxConcurrency) {
        NettyNioAsyncHttpClient.Builder httpClientBuilder = NettyNioAsyncHttpClient.builder()
            .protocol(Protocol.HTTP2)
            .maxConcurrency(effectiveMaxConcurrency)
            .connectionTimeout(Duration.ofSeconds(connectionTimeoutSec))
            .http2Configuration(
                Http2Configuration.builder()
                    .maxStreams((long) maxStreamsPerConnection)
                    .healthCheckPingPeriod(Duration.ofSeconds(5))
                    .build()
            );

        return SageMakerRuntimeHttp2AsyncClient.builder()
            .region(Region.of(region))
            .httpClientBuilder(httpClientBuilder)
            .endpointOverride(URI.create(
                "https://runtime.sagemaker." + region + ".amazonaws.com:8443"))
            .build();
    }

    private String buildQueryString() {
        Map<String, String> params = new LinkedHashMap<>();
        params.put("model", model);
        params.put("language", language);
        params.put("encoding", "linear16");
        params.put("punctuate", String.valueOf(punctuate));
        params.put("diarize", String.valueOf(diarize));
        params.put("interim_results", String.valueOf(interimResults));
        if (smartFormat) params.put("smart_format", "true");

        // Read sample rate from WAV
        try (AudioInputStream ais = AudioSystem.getAudioInputStream(wavFile.toFile())) {
            params.put("sample_rate", String.valueOf((int) ais.getFormat().getSampleRate()));
            params.put("channels", String.valueOf(ais.getFormat().getChannels()));
        } catch (Exception e) {
            log.warn("Could not read WAV sample rate, using defaults");
            params.put("sample_rate", "16000");
            params.put("channels", "1");
        }

        StringBuilder qs = new StringBuilder();
        params.forEach((k, v) -> {
            if (qs.length() > 0) qs.append("&");
            qs.append(k).append("=").append(v);
        });

        if (keyterms != null && !keyterms.isBlank()) {
            for (String kt : keyterms.split(",")) {
                qs.append("&keyterm=").append(kt.trim());
            }
        }

        return qs.toString();
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

    private void printStatus(List<StreamingConnection> conns, long testStartNanos) {
        if (conns.isEmpty()) return;
        long activeCount = conns.stream().filter(StreamingConnection::isActive).count();
        long errorCount = conns.stream().filter(StreamingConnection::isErrored).count();
        int totalTranscripts = conns.stream().mapToInt(StreamingConnection::getTranscriptCount).sum();
        int totalChunks = conns.stream().mapToInt(StreamingConnection::getChunkCount).sum();
        int totalRetries = conns.stream().mapToInt(StreamingConnection::getRetryCount).sum();
        double elapsed = (System.nanoTime() - testStartNanos) / 1_000_000_000.0;

        System.err.printf("\r[%s] Active: %d/%d | Errored: %d | Retries: %d | Transcripts: %d | Chunks: %d    ",
            formatDuration(elapsed), activeCount, conns.size(), errorCount, totalRetries, totalTranscripts, totalChunks);
    }

    private void printSummary(List<StreamingConnection> conns, double wallTime) {
        System.err.println();
        System.err.println();
        System.err.println("=== STREAM SUMMARY ===");

        long successful = conns.stream().filter(c -> !c.isErrored()).count();
        long errored = conns.stream().filter(StreamingConnection::isErrored).count();
        int totalTranscripts = conns.stream().mapToInt(StreamingConnection::getTranscriptCount).sum();
        int totalChunks = conns.stream().mapToInt(StreamingConnection::getChunkCount).sum();
        int totalRetries = conns.stream().mapToInt(StreamingConnection::getRetryCount).sum();

        System.err.println("Total connections:  " + conns.size());
        System.err.println("Successful:         " + successful);
        System.err.println("Errored:            " + errored);
        System.err.println("Total retries:      " + totalRetries);
        System.err.println("Total transcripts:  " + totalTranscripts);
        System.err.println("Total chunks sent:  " + totalChunks);
        System.err.printf("Wall time:          %.2fs%n", wallTime);
        System.err.println();

        // Duration stats for successful connections
        List<Double> durations = conns.stream()
            .filter(c -> !c.isErrored())
            .map(StreamingConnection::getDurationSeconds)
            .sorted()
            .collect(Collectors.toList());

        if (!durations.isEmpty()) {
            System.err.println("--- Connection Durations (successful) ---");
            System.err.printf("Min:    %.2fs%n", durations.get(0));
            System.err.printf("Median: %.2fs%n", durations.get(durations.size() / 2));
            System.err.printf("Max:    %.2fs%n", durations.get(durations.size() - 1));
            System.err.printf("Mean:   %.2fs%n", durations.stream().mapToDouble(d -> d).average().orElse(0));
        }

        // Error breakdown
        if (errored > 0) {
            System.err.println();
            System.err.println("--- Errors ---");
            Map<String, Long> errorCounts = conns.stream()
                .filter(StreamingConnection::isErrored)
                .flatMap(c -> c.getErrorMessages().stream())
                .collect(Collectors.groupingBy(m -> truncate(m, 120), Collectors.counting()));
            errorCounts.entrySet().stream()
                .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
                .forEach(e -> System.err.printf("  [%dx] %s%n", e.getValue(), e.getKey()));
        }

        System.err.println();
        System.err.println("--- HTTP/2 Configuration Used ---");
        System.err.println("maxStreamsPerConnection: " + maxStreamsPerConnection);
        System.err.println("maxConcurrency:         " + (maxConcurrency > 0 ? maxConcurrency : connections + 50));
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
        ch.qos.logback.classic.Logger root =
            (ch.qos.logback.classic.Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
        ch.qos.logback.classic.Logger appLogger =
            (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("com.deepgram.loadtest");

        switch (logLevel.toUpperCase()) {
            case "DEBUG":
                appLogger.setLevel(ch.qos.logback.classic.Level.DEBUG);
                break;
            case "WARN":
                appLogger.setLevel(ch.qos.logback.classic.Level.WARN);
                break;
            case "ERROR":
                appLogger.setLevel(ch.qos.logback.classic.Level.ERROR);
                break;
            default:
                appLogger.setLevel(ch.qos.logback.classic.Level.INFO);
                break;
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
