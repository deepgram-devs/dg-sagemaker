package com.deepgram.loadtest;

import com.deepgram.DeepgramClient;
import com.deepgram.resources.listen.v1.types.ListenV1CloseStream;
import com.deepgram.resources.listen.v1.types.ListenV1CloseStreamType;
import com.deepgram.resources.listen.v1.websocket.V1ConnectOptions;
import com.deepgram.resources.listen.v1.websocket.V1WebSocketClient;
import com.deepgram.sagemaker.SageMakerConfig;
import com.deepgram.sagemaker.SageMakerTransportFactory;

import okio.ByteString;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A single streaming connection using the Deepgram Java SDK + SageMaker transport.
 *
 * Path through the stack:
 *   DeepgramClient -> SageMakerTransportFactory -> SageMakerTransport -> AWS SDK HTTP/2
 *
 * Each connection creates its own SageMakerTransportFactory (and therefore its
 * own SageMakerRuntimeHttp2AsyncClient) to ensure stream isolation.
 */
public class SdkStreamingConnection {

    private static final Logger log = LoggerFactory.getLogger(SdkStreamingConnection.class);
    private static final int CHUNK_SIZE = 8192;
    // If real-time pacing drifts more than this far behind, reset the baseline so the loop
    // doesn't burst-send the catch-up audio at line speed (which overruns the model and
    // truncates tail-end transcripts after CloseStream).
    private static final long PACING_DRIFT_THRESHOLD_NANOS = 1_000_000_000L;  // 1 s

    private final int connectionId;
    private final String endpointName;
    private final String region;
    private final Path wavPath;
    private final V1ConnectOptions connectOptions;
    private final boolean loop;
    private final int maxRetries;
    private final int awaitFinalResultsSeconds;
    private final Runnable onFirstPassComplete;

    // Metrics
    private final AtomicInteger chunkCount = new AtomicInteger();
    private final AtomicInteger transcriptCount = new AtomicInteger();
    private final AtomicInteger retryCount = new AtomicInteger();
    private final AtomicBoolean errored = new AtomicBoolean(false);
    private final List<String> errorMessages = Collections.synchronizedList(new ArrayList<>());
    private final AtomicBoolean active = new AtomicBoolean(false);
    private final AtomicBoolean stopped = new AtomicBoolean(false);
    private final AtomicBoolean firstPassFired = new AtomicBoolean(false);

    // Timing
    private final AtomicLong startTimeNanos = new AtomicLong();
    private final AtomicLong endTimeNanos = new AtomicLong();

    public SdkStreamingConnection(int connectionId, String endpointName, String region,
                                  Path wavPath, V1ConnectOptions connectOptions,
                                  boolean loop, int maxRetries, int awaitFinalResultsSeconds,
                                  Runnable onFirstPassComplete) {
        this.connectionId = connectionId;
        this.endpointName = endpointName;
        this.region = region;
        this.wavPath = wavPath;
        this.connectOptions = connectOptions;
        this.loop = loop;
        this.maxRetries = maxRetries;
        this.awaitFinalResultsSeconds = awaitFinalResultsSeconds;
        this.onFirstPassComplete = onFirstPassComplete;
    }

    public CompletableFuture<Void> start() {
        startTimeNanos.set(System.nanoTime());
        CompletableFuture<Void> result = new CompletableFuture<>();

        Thread retryThread = new Thread(() -> runWithRetry(result), "sdk-conn-" + connectionId);
        retryThread.setDaemon(true);
        retryThread.start();

        return result;
    }

    private void runWithRetry(CompletableFuture<Void> result) {
        int attempt = 0;
        active.set(true);

        while (!stopped.get()) {
            SageMakerTransportFactory factory = null;
            try {
                // Each attempt gets its own factory -> its own AWS SDK client
                SageMakerConfig smConfig = SageMakerConfig.builder()
                        .endpointName(endpointName)
                        .region(region)
                        .build();
                factory = new SageMakerTransportFactory(smConfig);

                DeepgramClient client = DeepgramClient.builder()
                        .apiKey("unused")
                        .transportFactory(factory)
                        .build();

                V1WebSocketClient wsClient = client.listen().v1().v1WebSocket();
                CountDownLatch done = new CountDownLatch(1);
                AtomicBoolean streamErrored = new AtomicBoolean(false);
                List<String> streamErrors = Collections.synchronizedList(new ArrayList<>());

                wsClient.onResults(r -> {
                    if (r.getChannel() != null
                            && r.getChannel().getAlternatives() != null
                            && !r.getChannel().getAlternatives().isEmpty()) {
                        String transcript = r.getChannel().getAlternatives().get(0).getTranscript();
                        boolean isFinal = r.getIsFinal().orElse(false);
                        if (isFinal && transcript != null && !transcript.isEmpty()) {
                            int count = transcriptCount.incrementAndGet();
                            log.info("[Conn {}] #{}: {}", connectionId, count, transcript);
                        }
                    }
                });

                wsClient.onError(error -> {
                    streamErrored.set(true);
                    streamErrors.add(error.getMessage());
                    done.countDown();
                });

                wsClient.onDisconnected(reason -> {
                    log.debug("[Conn {}] Disconnected (code {})", connectionId, reason.getCode());
                    done.countDown();
                });

                // Connect
                wsClient.connect(connectOptions).get(30, TimeUnit.SECONDS);
                log.debug("[Conn {}] Connected", connectionId);

                // Stream audio
                streamAudio(wsClient);

                // Signal end of audio
                if (!stopped.get()) {
                    wsClient.sendCloseStream(
                            ListenV1CloseStream.builder()
                                    .type(ListenV1CloseStreamType.CLOSE_STREAM)
                                    .build());
                }

                // Wait for final results
                done.await(awaitFinalResultsSeconds, TimeUnit.SECONDS);
                wsClient.disconnect();

                if (streamErrored.get()) {
                    throw new RuntimeException(String.join("; ", streamErrors));
                }

                // Success
                endTimeNanos.compareAndSet(0, System.nanoTime());

                if (!loop || stopped.get()) {
                    active.set(false);
                    result.complete(null);
                    return;
                }
                attempt = 0;

            } catch (Exception e) {
                String msg = e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName();

                if (stopped.get()) {
                    active.set(false);
                    result.complete(null);
                    return;
                }

                boolean retryable = isRetryable(msg);
                if (retryable && attempt < maxRetries) {
                    attempt++;
                    retryCount.incrementAndGet();
                    long waitMs = Math.min(1000L * (1L << (attempt - 1)), 30_000);
                    log.warn("[Conn {}] Retryable error (attempt {}/{}), retrying in {}ms: {}",
                            connectionId, attempt, maxRetries, waitMs, truncate(msg, 100));
                    try { Thread.sleep(waitMs); } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        active.set(false);
                        result.complete(null);
                        return;
                    }
                    continue;
                }

                active.set(false);
                errored.set(true);
                errorMessages.add(msg);
                endTimeNanos.set(System.nanoTime());
                if (attempt >= maxRetries) {
                    log.error("[Conn {}] Retries exhausted ({}/{}): {}", connectionId, attempt, maxRetries, truncate(msg, 100));
                } else {
                    log.error("[Conn {}] Non-retryable error: {}", connectionId, truncate(msg, 100));
                }
                result.complete(null);
                return;
            } finally {
                if (factory != null) {
                    try { factory.shutdown(); } catch (Exception ignore) {}
                }
            }
        }
        active.set(false);
        result.complete(null);
    }

    private void streamAudio(V1WebSocketClient wsClient) throws Exception {
        int totalChunks = chunkCount.get();
        int playCount = 0;
        long streamStartNanos = System.nanoTime();
        // Anchor for the pacing math: starts at totalChunks at loop entry, advances on
        // each pacing reset so target_audio_secs is always measured relative to the most
        // recent reset rather than the absolute beginning of the stream.
        int chunksAtPacingStart = totalChunks;

        while (!stopped.get()) {
            playCount++;
            log.debug("[Conn {}] Streaming WAV (pass {})...", connectionId, playCount);

            try (AudioInputStream ais = AudioSystem.getAudioInputStream(wavPath.toFile())) {
                AudioFormat fmt = ais.getFormat();
                int bytesPerFrame = fmt.getFrameSize();
                int framesPerChunk = CHUNK_SIZE / bytesPerFrame;
                double chunkDurationSec = (double) framesPerChunk / fmt.getSampleRate();

                byte[] buf = new byte[CHUNK_SIZE];
                int bytesRead;

                while (!stopped.get() && (bytesRead = ais.read(buf, 0, CHUNK_SIZE)) > 0) {
                    byte[] chunk = (bytesRead == CHUNK_SIZE)
                            ? buf.clone()
                            : java.util.Arrays.copyOf(buf, bytesRead);
                    wsClient.sendMedia(ByteString.of(chunk));
                    totalChunks++;
                    chunkCount.set(totalChunks);

                    // Pace to real-time, with a drift-reset guard. If sendMedia blocked for
                    // a long time (SDK retry storm, throttle backoff), the naive pacing math
                    // sees `elapsed >> targetSec` and burst-sends the next N chunks at line
                    // speed to "catch up". That overruns the model's input buffer with a
                    // backlog that can't be flushed in the post-CloseStream window, so
                    // tail-end transcripts get truncated and WER inflates. When drift exceeds
                    // PACING_DRIFT_THRESHOLD_NANOS, reset the baseline so pacing resumes at
                    // real-time from now. The conn's wall-clock runtime grows by the storm
                    // duration but the model never sees a backlog.
                    long elapsedNanos = System.nanoTime() - streamStartNanos;
                    double targetSec = (totalChunks - chunksAtPacingStart) * chunkDurationSec;
                    long sleepNanos = (long) (targetSec * 1_000_000_000L) - elapsedNanos;
                    if (sleepNanos < -PACING_DRIFT_THRESHOLD_NANOS) {
                        log.info("[Conn {}] pacing drift {}ms detected — resetting baseline",
                                connectionId, -sleepNanos / 1_000_000);
                        streamStartNanos = System.nanoTime();
                        chunksAtPacingStart = totalChunks;
                        sleepNanos = 0;
                    }
                    if (sleepNanos > 0) {
                        Thread.sleep(sleepNanos / 1_000_000, (int) (sleepNanos % 1_000_000));
                    }
                }
            }

            // One full pass of the file completed (reached EOF, not stopped mid-file)
            if (!stopped.get() && firstPassFired.compareAndSet(false, true)
                    && onFirstPassComplete != null) {
                onFirstPassComplete.run();
            }

            if (!loop) break;
        }

        log.info("[Conn {}] Audio streaming done ({} passes, {} chunks)", connectionId, playCount, totalChunks);
    }

    private static boolean isRetryable(String msg) {
        if (msg == null) return false;
        return msg.contains("ThrottlingException")
                || msg.contains("ModelStreamError")
                || msg.contains("ServiceUnavailable")
                || msg.contains("InternalStreamFailure")
                || msg.contains("Unable to execute HTTP request")
                || msg.contains("connection was closed")
                || msg.contains("Acquire operation took longer");
    }

    public void stop() {
        stopped.set(true);
        active.set(false);
    }

    // --- Accessors ---
    public int getConnectionId() { return connectionId; }
    public int getChunkCount() { return chunkCount.get(); }
    public int getTranscriptCount() { return transcriptCount.get(); }
    public int getRetryCount() { return retryCount.get(); }
    public boolean isErrored() { return errored.get(); }
    public List<String> getErrorMessages() { return errorMessages; }
    public boolean isActive() { return active.get(); }

    public double getDurationSeconds() {
        long start = startTimeNanos.get();
        long end = endTimeNanos.get();
        if (start == 0) return 0;
        if (end == 0) end = System.nanoTime();
        return (end - start) / 1_000_000_000.0;
    }

    private static String truncate(String s, int maxLen) {
        return (s != null && s.length() > maxLen) ? s.substring(0, maxLen) + "..." : s;
    }
}
