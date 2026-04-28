package com.deepgram.loadtest;

import org.reactivestreams.Publisher;
import org.reactivestreams.Subscriber;
import org.reactivestreams.Subscription;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.awssdk.core.SdkBytes;
import software.amazon.awssdk.services.sagemakerruntimehttp2.SageMakerRuntimeHttp2AsyncClient;
import software.amazon.awssdk.services.sagemakerruntimehttp2.model.*;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;

/**
 * A single bidirectional streaming connection to a Deepgram model on SageMaker.
 *
 * Each connection:
 *  - Opens one HTTP/2 bidi stream via InvokeEndpointWithBidirectionalStream
 *  - Streams WAV audio paced to real-time speed
 *  - Receives and counts transcript responses
 *  - Tracks per-connection metrics (chunks sent, transcripts received, errors)
 */
public class StreamingConnection {

    private static final Logger log = LoggerFactory.getLogger(StreamingConnection.class);
    private static final int CHUNK_SIZE = 8192;

    // Deepgram CloseStream control message
    // (see https://developers.deepgram.com/docs/close-stream)
    private static final byte[] CLOSE_STREAM_BYTES =
        "{\"type\":\"CloseStream\"}".getBytes(java.nio.charset.StandardCharsets.UTF_8);

    private final int connectionId;
    private final SageMakerRuntimeHttp2AsyncClient client;
    private final String endpointName;
    private final Path wavPath;
    private final String queryString;
    private final boolean loop;
    private final int maxRetries;
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

    // Streaming state
    private volatile CompletableFuture<Void> streamFuture;
    private volatile AudioPublisher audioPublisher;

    // Timing
    private final AtomicLong startTimeNanos = new AtomicLong();
    private final AtomicLong endTimeNanos = new AtomicLong();

    public StreamingConnection(int connectionId, SageMakerRuntimeHttp2AsyncClient client,
                               String endpointName, Path wavPath, String queryString,
                               boolean loop, int maxRetries,
                               Runnable onFirstPassComplete) {
        this.connectionId = connectionId;
        this.client = client;
        this.endpointName = endpointName;
        this.wavPath = wavPath;
        this.queryString = queryString;
        this.loop = loop;
        this.maxRetries = maxRetries;
        this.onFirstPassComplete = onFirstPassComplete;
    }

    /**
     * Start the bidirectional streaming session with retry logic.
     * On retryable errors (ThrottlingException, ModelStreamError), waits with
     * exponential backoff and retries up to maxRetries times.
     * Returns a future that completes when streaming finishes or retries are exhausted.
     */
    public CompletableFuture<Void> start() {
        startTimeNanos.set(System.nanoTime());
        CompletableFuture<Void> result = new CompletableFuture<>();

        // Run the retry loop on a daemon thread
        Thread retryThread = new Thread(() -> runWithRetry(result), "retry-conn-" + connectionId);
        retryThread.setDaemon(true);
        retryThread.start();

        return result;
    }

    private void runWithRetry(CompletableFuture<Void> result) {
        int attempt = 0;
        active.set(true);
        while (!stopped.get()) {
            try {
                // Reset per-attempt state (active stays true through retries)
                errored.set(false);
                audioPublisher = new AudioPublisher();

                CompletableFuture<Void> streamDone = startOneStream();

                // Start audio on a separate thread
                Thread audioThread = new Thread(this::streamAudio, "audio-conn-" + connectionId);
                audioThread.setDaemon(true);
                audioThread.start();

                // Block until this stream completes or errors
                streamDone.join();

                // Stream completed successfully
                endTimeNanos.compareAndSet(0, System.nanoTime());

                if (!loop || stopped.get()) {
                    active.set(false);
                    result.complete(null);
                    return;
                }
                // Looping — reset attempt counter on success and start a new stream
                attempt = 0;

            } catch (Exception e) {
                String msg = unwrapMessage(e);

                if (stopped.get()) {
                    active.set(false);
                    result.complete(null);
                    return;
                }

                boolean retryable = isRetryable(msg);
                if (retryable && attempt < maxRetries) {
                    attempt++;
                    retryCount.incrementAndGet();
                    long waitMs = Math.min(1000L * (1L << (attempt - 1)), 30_000); // 1s, 2s, 4s, ... capped at 30s
                    log.warn("[Conn {}] Retryable error (attempt {}/{}), retrying in {}ms: {}",
                        connectionId, attempt, maxRetries, waitMs, truncate(msg, 100));
                    try {
                        Thread.sleep(waitMs);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        active.set(false);
                        result.complete(null);
                        return;
                    }
                    continue;
                }

                // Non-retryable or retries exhausted
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
            }
        }
        active.set(false);
        result.complete(null);
    }

    private CompletableFuture<Void> startOneStream() {
        InvokeEndpointWithBidirectionalStreamRequest request =
            InvokeEndpointWithBidirectionalStreamRequest.builder()
                .endpointName(endpointName)
                .modelInvocationPath("v1/listen")
                .modelQueryString(queryString)
                .build();

        CompletableFuture<Void> errorFuture = new CompletableFuture<>();

        InvokeEndpointWithBidirectionalStreamResponseHandler responseHandler =
            InvokeEndpointWithBidirectionalStreamResponseHandler.builder()
                .onResponse(r -> log.debug("[Conn {}] Initial response received", connectionId))
                .onError(t -> {
                    errorFuture.completeExceptionally(t);
                })
                .onComplete(() -> {
                    log.debug("[Conn {}] Stream complete", connectionId);
                })
                .subscriber(InvokeEndpointWithBidirectionalStreamResponseHandler.Visitor.builder()
                    .onPayloadPart(payloadPart -> handleResponse(payloadPart))
                    .onDefault(event -> log.debug("[Conn {}] Unknown response event type", connectionId))
                    .build())
                .build();

        streamFuture = client.invokeEndpointWithBidirectionalStream(
            request, audioPublisher, responseHandler
        );

        // Return whichever completes first — the stream future or an error
        return CompletableFuture.anyOf(streamFuture, errorFuture).thenApply(v -> null);
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

    private static String unwrapMessage(Throwable t) {
        Throwable cause = t;
        while (cause.getCause() != null && cause.getCause() != cause) {
            cause = cause.getCause();
        }
        String msg = cause.getMessage();
        if (msg == null) msg = cause.getClass().getSimpleName();
        // Also check the original exception message
        String outerMsg = t.getMessage();
        if (outerMsg != null && outerMsg.length() > msg.length()) {
            return outerMsg;
        }
        return msg;
    }

    private static String truncate(String s, int maxLen) {
        return (s != null && s.length() > maxLen) ? s.substring(0, maxLen) + "..." : s;
    }

    private void handleResponse(ResponsePayloadPart payloadPart) {
        if (payloadPart.bytes() == null) return;
        String raw = payloadPart.bytes().asUtf8String();

        try {
            // Simple JSON parsing without pulling in a JSON library
            if (!raw.contains("\"channel\"")) {
                // Check for endpoint error
                if (raw.contains("\"error\"") || raw.contains("\"message\"")) {
                    errored.set(true);
                    errorMessages.add(raw.trim());
                    log.error("[Conn {}] Endpoint error: {}", connectionId, raw.trim());
                }
                return;
            }

            // Count final transcripts
            if (raw.contains("\"is_final\":true") || raw.contains("\"is_final\": true")) {
                int count = transcriptCount.incrementAndGet();
                // Extract transcript text for logging (best-effort)
                int idx = raw.indexOf("\"transcript\":\"");
                if (idx >= 0) {
                    int start = idx + 14;
                    int end = raw.indexOf("\"", start);
                    if (end > start) {
                        String transcript = raw.substring(start, end);
                        if (!transcript.isEmpty()) {
                            log.info("[Conn {}] #{}: {}", connectionId, count, transcript);
                        }
                    }
                }
            }
        } catch (Exception e) {
            log.warn("[Conn {}] Error parsing response: {}", connectionId, e.getMessage());
        }
    }

    /**
     * Stream WAV audio to the publisher, paced to real-time speed.
     */
    private void streamAudio() {
        int totalChunks = 0;
        int playCount = 0;
        long streamStartNanos = System.nanoTime();

        try {
            while (active.get()) {
                playCount++;
                log.debug("[Conn {}] Streaming WAV (pass {})...", connectionId, playCount);

                try (AudioInputStream ais = AudioSystem.getAudioInputStream(wavPath.toFile())) {
                    AudioFormat fmt = ais.getFormat();
                    int bytesPerFrame = fmt.getFrameSize();
                    int framesPerChunk = CHUNK_SIZE / bytesPerFrame;
                    double chunkDurationSec = (double) framesPerChunk / fmt.getSampleRate();

                    byte[] buf = new byte[CHUNK_SIZE];
                    int bytesRead;

                    while (active.get() && (bytesRead = ais.read(buf, 0, CHUNK_SIZE)) > 0) {
                        byte[] chunk = (bytesRead == CHUNK_SIZE) ? buf.clone() : java.util.Arrays.copyOf(buf, bytesRead);
                        audioPublisher.publish(chunk);
                        totalChunks++;
                        chunkCount.set(totalChunks);

                        // Pace to real-time
                        long elapsedNanos = System.nanoTime() - streamStartNanos;
                        double targetSec = totalChunks * chunkDurationSec;
                        long sleepNanos = (long) (targetSec * 1_000_000_000L) - elapsedNanos;
                        if (sleepNanos > 0) {
                            Thread.sleep(sleepNanos / 1_000_000, (int) (sleepNanos % 1_000_000));
                        }
                    }
                }

                // One full pass of the file completed (reached EOF, not stopped mid-file)
                if (active.get() && firstPassFired.compareAndSet(false, true)
                        && onFirstPassComplete != null) {
                    onFirstPassComplete.run();
                }

                if (!loop) break;
            }
        } catch (UnsupportedAudioFileException | IOException e) {
            log.error("[Conn {}] Audio read error: {}", connectionId, e.getMessage());
            errored.set(true);
            errorMessages.add("Audio error: " + e.getMessage());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            // Send CloseStream so Deepgram flushes the final transcript before
            // closing (https://developers.deepgram.com/docs/close-stream).
            audioPublisher.publish(CLOSE_STREAM_BYTES);
            audioPublisher.complete();
            log.info("[Conn {}] Audio streaming done ({} passes, {} chunks)",
                connectionId, playCount, totalChunks);
        }
    }

    public void stop() {
        stopped.set(true);
        active.set(false);
        if (audioPublisher != null) {
            // Enqueue CloseStream BEFORE complete() so the streamAudio finally
            // fallback isn't the only path — graceful shutdown via stop() also
            // sends the documented end-of-stream signal.
            audioPublisher.publish(CLOSE_STREAM_BYTES);
            audioPublisher.complete();
        }
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

    // -----------------------------------------------------------------------
    // AudioPublisher: bridges imperative audio chunk writes to the reactive
    // Publisher<RequestStreamEvent> required by the AWS SDK.
    // -----------------------------------------------------------------------
    private static class AudioPublisher implements Publisher<RequestStreamEvent> {
        private volatile Subscriber<? super RequestStreamEvent> subscriber;
        private final AtomicBoolean completed = new AtomicBoolean(false);
        private final java.util.concurrent.LinkedBlockingQueue<byte[]> queue =
            new java.util.concurrent.LinkedBlockingQueue<>(512);

        @Override
        public void subscribe(Subscriber<? super RequestStreamEvent> s) {
            this.subscriber = s;
            s.onSubscribe(new Subscription() {
                @Override
                public void request(long n) {
                    // Drain queued chunks on a daemon thread
                    Thread drainThread = new Thread(() -> drain(n));
                    drainThread.setDaemon(true);
                    drainThread.start();
                }

                @Override
                public void cancel() {
                    completed.set(true);
                }
            });
        }

        private void drain(long requested) {
            long emitted = 0;
            // Keep draining queued chunks even after `completed` flips so a
            // trailing CloseStream message isn't dropped before it's emitted.
            while (emitted < requested) {
                try {
                    byte[] chunk = queue.poll(200, java.util.concurrent.TimeUnit.MILLISECONDS);
                    if (chunk == null) {
                        if (completed.get()) break;
                        continue;
                    }
                    RequestStreamEvent event = RequestStreamEvent.payloadPartBuilder()
                        .bytes(SdkBytes.fromByteArray(chunk))
                        .build();
                    subscriber.onNext(event);
                    emitted++;
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
            if (completed.get() && queue.isEmpty()) {
                subscriber.onComplete();
            }
        }

        void publish(byte[] chunk) {
            if (completed.get()) return;
            try {
                queue.put(chunk);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        void complete() {
            completed.set(true);
        }
    }
}
