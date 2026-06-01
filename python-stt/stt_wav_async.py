#!/usr/bin/env python3
"""
Deepgram SageMaker WAV File STT Client - Async Inference (S3 In/Out)

Transcribes a WAV file via the SageMaker `InvokeEndpointAsync` API on an
endpoint whose `EndpointConfig` carries an `AsyncInferenceConfig`. Unlike
the synchronous `batch` mode (capped at a 25 MB request body), async
inference accepts up to a 1 GB S3 object and a 60-minute processing budget,
making it the right path for long-form pre-recorded audio.

Flow
----
1.  Validate the local WAV file (must be 16-bit PCM).
2.  Upload the WAV to S3 (or use a pre-existing S3 URI with --input-s3-uri).
3.  Call `InvokeEndpointAsync` for each request, passing the model + language
    + feature flags through `CustomAttributes` formatted as
    `v1/listen?key=value&...`. The path prefix is required — without it
    the Deepgram container returns 404.
4.  Poll the success + failure S3 prefixes published by the EndpointConfig
    until each invocation lands an `<UUID>.out` or `<UUID>-error.out`.
5.  Download + parse the success payloads and print a per-request transcript
    summary plus an aggregate latency / throughput table.

Endpoint prerequisites
----------------------
- The endpoint must be created from an `EndpointConfig` that includes an
  `AsyncInferenceConfig` block with `OutputConfig.S3OutputPath` and
  `S3FailurePath`.
- The SageMaker execution role attached to the endpoint must have
  `s3:GetObject` on the input prefix, `s3:PutObject` on the output +
  failures prefixes, and `s3:ListBucket` on the bucket.
- `AsyncInferenceConfig` is incompatible with `EnableSSMAccess=true` on
  the same production variant — pick one or the other.

This script only needs `s3:PutObject` on the upload key (skipped when
`--input-s3-uri` is supplied), `s3:GetObject` on the success/failure keys,
and `sagemaker:InvokeEndpointAsync` on the endpoint.
"""

import argparse
import concurrent.futures
import json
import logging
import statistics
import sys
import threading
import time
import uuid
import wave
from urllib.parse import quote, urlparse

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

# Configuration constants
DEFAULT_REGION = "us-east-2"

# AsyncInferenceConfig limits, per AWS:
# https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html
ASYNC_INVOKE_MAX_BYTES = 1 * 1024 * 1024 * 1024  # 1 GiB
ASYNC_INVOKE_MAX_TIMEOUT_S = 3600  # 60 minutes

DEFAULT_INVOCATION_TIMEOUT_S = 3600
DEFAULT_POLL_INTERVAL_S = 5.0
DEFAULT_UPLOAD_PREFIX = "stt-async-input"  # under {bucket}/<this>/<uuid>.wav

# Deepgram supported redaction entity types
# https://developers.deepgram.com/docs/supported-entity-types
REDACT_ENTITIES = [
    # Broad group redactions
    "pii", "phi", "pci",
    # PII entities
    "account_number", "age", "bank_account", "cardinal", "credit_card",
    "credit_card_expiration", "cvv", "date", "date_interval", "dob",
    "driver_license", "email_address", "event", "filename", "gender_sexuality",
    "healthcare_number", "ip_address", "location", "location_address",
    "location_city", "location_coordinate", "location_country", "location_state",
    "location_zip", "money", "name", "name_given", "name_family",
    "name_medical_professional", "numerical_pii", "occupation", "ordinal",
    "origin", "passport_number", "password", "percent", "phone_number",
    "physical_attribute", "ssn", "time", "url", "username", "vehicle_id",
    # PHI entities
    "condition", "drug", "injury", "blood_type", "medical_process", "statistics",
    # Other entities
    "language", "marital_status", "organization", "political_affiliation",
    "religion", "routing_number", "zodiac_sign",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_redact(value: str) -> list[str]:
    """
    Parse a comma-separated --redact string into a validated list of entity types.
    """
    if not value:
        return []
    entities = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in REDACT_ENTITIES:
            print(
                f"ERROR: '{item}' is not a valid redact entity. "
                f"Supported values: {', '.join(REDACT_ENTITIES)}"
            )
            raise SystemExit(1)
        entities.append(item)
    return entities


def _parse_extra(value: str) -> list[tuple[str, str]]:
    """
    Parse `--extra k=v&k2=v2` into a list of (key, value) pairs preserving order.
    """
    if not value:
        return []
    out: list[tuple[str, str]] = []
    for pair in value.split("&"):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            print(f"ERROR: --extra entry '{pair}' must be in k=v form")
            raise SystemExit(1)
        k, v = pair.split("=", 1)
        out.append((k.strip(), v.strip()))
    return out


def _build_async_query_string(args, extra_pairs: list[tuple[str, str]]) -> str:
    """
    Build the Deepgram invocation path + query string for async inference.

    The string is sent as `X-Amzn-SageMaker-Custom-Attributes`. The
    Deepgram container parses it as `path?query`, so the leading
    `v1/listen?` prefix is mandatory — a bare query string lands on a
    non-routable path inside stem and returns 404.
    """
    params: list[str] = [
        f"model={args.model}",
        f"language={args.language}",
        f"punctuate={args.punctuate}",
        f"diarize={args.diarize}",
    ]
    if args.redact:
        params.extend(f"redact={quote(r)}" for r in args.redact)
    if args.keyterms:
        params.extend(f"keyterm={quote(kt)}" for kt in args.keyterms)
    for k, v in extra_pairs:
        params.append(f"{quote(k)}={quote(v)}")
    return "v1/listen?" + "&".join(params)


def _extract_transcript(result: dict) -> tuple[str, float, float]:
    """
    Pull (transcript, confidence, duration_seconds) from a Deepgram
    pre-recorded response. Returns ("", 0.0, 0.0) on unexpected shape.
    """
    try:
        alt = result["results"]["channels"][0]["alternatives"][0]
    except (KeyError, IndexError, TypeError):
        return "", 0.0, 0.0
    transcript = alt.get("transcript", "") or ""
    confidence = float(alt.get("confidence", 0.0) or 0.0)
    duration = float(((result or {}).get("metadata") or {}).get("duration", 0.0) or 0.0)
    return transcript, confidence, duration


def _split_s3_uri(s3_uri: str) -> tuple[str, str]:
    """
    Split `s3://bucket/key/path` into (bucket, key). Raises on malformed URIs.
    """
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(
            f"Expected an s3:// URI, got: {s3_uri!r}"
        )
    return parsed.netloc, parsed.path.lstrip("/")


def _validate_wav(wav_path: str) -> tuple[int, int, float, int]:
    """
    Validate a 16-bit PCM WAV file and return (sample_rate, channels,
    duration_s, size_bytes). Raises ValueError for non-PCM-16 or
    over-1-GiB inputs.
    """
    with wave.open(wav_path, "rb") as wf:
        sample_width = wf.getsampwidth()
        if sample_width != 2:
            raise ValueError(
                f"WAV file must be 16-bit PCM (sample width 2 bytes). "
                f"Got {sample_width * 8}-bit audio. "
                "Convert with: ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 output.wav"
            )
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        duration = wf.getnframes() / sample_rate

    import os
    size_bytes = os.path.getsize(wav_path)
    if size_bytes > ASYNC_INVOKE_MAX_BYTES:
        gb = size_bytes / (1024 ** 3)
        raise ValueError(
            f"WAV file is {gb:.2f} GiB, exceeds the SageMaker async "
            f"InvokeEndpoint 1 GiB body limit. Split the file with: "
            "ffmpeg -i input.wav -f segment -segment_time 1800 segment_%03d.wav"
        )
    return sample_rate, channels, duration, size_bytes


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class AsyncSTTClient:
    """
    Submits WAV transcription jobs to an async-configured SageMaker endpoint
    and collects the resulting S3 outputs.

    Each invocation runs on its own thread with its own boto3 client so
    concurrent submissions do not share connection state. Polling is also
    threaded; each in-flight invocation independently waits for its
    `<UUID>.out` or `<UUID>-error.out` to appear under the EndpointConfig's
    `S3OutputPath` / `S3FailurePath`.
    """

    def __init__(
        self,
        endpoint_name: str,
        region: str = DEFAULT_REGION,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        invocation_timeout_s: int = DEFAULT_INVOCATION_TIMEOUT_S,
    ):
        self.endpoint_name = endpoint_name
        self.region = region
        self.poll_interval_s = poll_interval_s
        self.invocation_timeout_s = invocation_timeout_s
        self._session: boto3.Session | None = None
        self._thread_local = threading.local()

    def _initialize_client(self):
        """Resolve AWS credentials and verify identity once for the whole run."""
        try:
            session = boto3.Session(region_name=self.region)
            credentials = session.get_credentials()
            if credentials is None:
                raise NoCredentialsError()
            credentials.get_frozen_credentials()

            caller_identity = session.client("sts").get_caller_identity()
            logger.debug(f"Authenticated as: {caller_identity.get('Arn', 'Unknown')}")

            self._session = session
            logger.info("AWS credentials resolved")

        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error("AWS credentials not found. Configure via one of:")
            logger.error("  1. aws configure")
            logger.error("  2. Environment variables: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY")
            logger.error("  3. ~/.aws/credentials")
            logger.error("  4. IAM role (when running on AWS infrastructure)")
            raise RuntimeError("AWS credentials not available") from e

    def _get_thread_clients(self):
        """Per-thread (s3, sagemaker-runtime) clients."""
        if not hasattr(self._thread_local, "clients"):
            self._thread_local.clients = (
                self._session.client("s3"),
                self._session.client("sagemaker-runtime"),
            )
        return self._thread_local.clients

    def upload_wav(self, wav_path: str, bucket: str, key: str) -> str:
        """
        Upload `wav_path` to `s3://bucket/key`. Returns the resulting S3 URI.
        """
        if self._session is None:
            self._initialize_client()
        s3, _ = self._get_thread_clients()
        s3.upload_file(
            Filename=wav_path,
            Bucket=bucket,
            Key=key,
            ExtraArgs={"ContentType": "audio/wav"},
        )
        return f"s3://{bucket}/{key}"

    def _invoke_once(
        self,
        request_id: int,
        input_s3_uri: str,
        custom_attributes: str,
        inference_id_prefix: str,
        stop_event: threading.Event,
    ) -> tuple[int, float, dict | None, str | None, Exception | None]:
        """
        Submit one async invocation and poll its result. Returns
        (request_id, elapsed_s, parsed_response_or_None, output_s3_uri_or_None,
        error_or_None).
        """
        if stop_event.is_set():
            return request_id, 0.0, None, None, RuntimeError("Aborted due to prior error")

        s3, sm_rt = self._get_thread_clients()
        start = time.monotonic()
        inf_id = f"{inference_id_prefix}-{request_id:03d}"

        try:
            response = sm_rt.invoke_endpoint_async(
                EndpointName=self.endpoint_name,
                InputLocation=input_s3_uri,
                ContentType="audio/wav",
                Accept="application/json",
                CustomAttributes=custom_attributes,
                InvocationTimeoutSeconds=self.invocation_timeout_s,
                InferenceId=inf_id,
            )
        except ClientError as e:
            elapsed = time.monotonic() - start
            stop_event.set()
            code = e.response["Error"]["Code"]
            msg = e.response["Error"]["Message"]
            return request_id, elapsed, None, None, RuntimeError(
                f"InvokeEndpointAsync failed [{code}]: {msg}"
            )

        output_uri = response["OutputLocation"]
        failure_uri = response["FailureLocation"]
        out_bucket, out_key = _split_s3_uri(output_uri)
        fail_bucket, fail_key = _split_s3_uri(failure_uri)

        deadline = start + self.invocation_timeout_s + 60  # +60s for S3 propagation
        while time.monotonic() < deadline:
            if stop_event.is_set():
                return request_id, time.monotonic() - start, None, None, RuntimeError(
                    "Aborted due to prior error"
                )

            # Probe both prefixes; first one wins.
            success = self._head_or_none(s3, out_bucket, out_key)
            if success is not None:
                elapsed = time.monotonic() - start
                body = s3.get_object(Bucket=out_bucket, Key=out_key)["Body"].read()
                try:
                    parsed = json.loads(body)
                except json.JSONDecodeError as e:
                    return request_id, elapsed, None, output_uri, RuntimeError(
                        f"Output is not JSON ({len(body)} bytes): {e}"
                    )
                return request_id, elapsed, parsed, output_uri, None

            failure = self._head_or_none(s3, fail_bucket, fail_key)
            if failure is not None:
                elapsed = time.monotonic() - start
                body = s3.get_object(Bucket=fail_bucket, Key=fail_key)["Body"].read()
                msg = body.decode("utf-8", errors="replace").strip()
                return request_id, elapsed, None, failure_uri, RuntimeError(
                    f"Endpoint reported failure: {msg}"
                )

            time.sleep(self.poll_interval_s)

        # Timed out waiting for either side.
        elapsed = time.monotonic() - start
        return request_id, elapsed, None, None, RuntimeError(
            f"Timed out after {elapsed:.0f}s waiting for {output_uri} / {failure_uri}"
        )

    @staticmethod
    def _head_or_none(s3, bucket: str, key: str) -> dict | None:
        try:
            return s3.head_object(Bucket=bucket, Key=key)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                return None
            raise

    def run(
        self,
        input_s3_uri: str,
        custom_attributes: str,
        inference_id_prefix: str,
        num_requests: int = 1,
        concurrency: int = 1,
    ) -> list[tuple[int, float, dict | None, str | None, Exception | None]]:
        """
        Submit `num_requests` async invocations with up to `concurrency` in flight.

        Returns a list of (request_id, elapsed_s, parsed_response, output_s3_uri,
        error) tuples in completion order.
        """
        if self._session is None:
            self._initialize_client()

        stop_event = threading.Event()
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix="async-invoke",
        )
        try:
            futures = {
                executor.submit(
                    self._invoke_once,
                    i + 1,
                    input_s3_uri,
                    custom_attributes,
                    inference_id_prefix,
                    stop_event,
                ): i + 1
                for i in range(num_requests)
            }
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    req_id = futures[future]
                    logger.error(f"[Request {req_id}] Unexpected thread error: {e}")
                    stop_event.set()
                    results.append((req_id, 0.0, None, None, e))
        except (KeyboardInterrupt, SystemExit):
            logger.warning("Async run interrupted — cancelling pending requests")
            stop_event.set()
            for f in futures:
                f.cancel()
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe a WAV file via the SageMaker `InvokeEndpointAsync` API "
            "(S3 input/output). Suitable for files larger than the 25 MB "
            "synchronous invoke limit (up to 1 GiB)."
        )
    )
    parser.add_argument("endpoint_name", help="SageMaker endpoint name (async-configured)")
    parser.add_argument(
        "--file",
        required=True,
        metavar="WAV_FILE",
        help="Path to a 16-bit PCM WAV file (up to 1 GiB).",
    )

    bucket_group = parser.add_mutually_exclusive_group(required=True)
    bucket_group.add_argument(
        "--bucket",
        help=(
            "S3 bucket to upload the WAV to. The script writes the object at "
            f"`s3://<bucket>/{DEFAULT_UPLOAD_PREFIX}/<uuid>.wav` and uses that "
            "as the async input location. Either --bucket or --input-s3-uri is "
            "required."
        ),
    )
    bucket_group.add_argument(
        "--input-s3-uri",
        metavar="S3_URI",
        help=(
            "Skip upload and use an existing S3 object as input. Must be an "
            "s3:// URI to a WAV file the endpoint's execution role can read."
        ),
    )
    parser.add_argument(
        "--upload-prefix",
        default=DEFAULT_UPLOAD_PREFIX,
        metavar="PREFIX",
        help=(
            f"Key prefix used when --bucket is set (default: "
            f"{DEFAULT_UPLOAD_PREFIX!r}). The full key is `<prefix>/<uuid>.wav`."
        ),
    )

    parser.add_argument(
        "--model",
        default="nova-3",
        help="Deepgram model (default: nova-3)",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code (default: en)",
    )
    parser.add_argument(
        "--punctuate",
        default="true",
        choices=["true", "false"],
        help="Enable punctuation (default: true)",
    )
    parser.add_argument(
        "--diarize",
        default="false",
        choices=["true", "false"],
        help="Enable speaker diarization (default: false)",
    )
    parser.add_argument(
        "--redact",
        default="",
        metavar="ENTITY,ENTITY,...",
        help=(
            "Comma-separated list of entity types to redact from transcripts. "
            f"Supported values: {', '.join(REDACT_ENTITIES)}"
        ),
    )
    parser.add_argument(
        "--keyterms",
        default="",
        metavar="TERM,TERM,...",
        help=(
            "Comma-separated keyterms to boost recognition (nova-3). "
            "Each term is sent as keyterm=<value>."
        ),
    )
    parser.add_argument(
        "--extra",
        default="",
        metavar="k=v&k2=v2",
        help=(
            "Extra Deepgram query parameters appended verbatim "
            "(e.g. 'sentiment=true&topics=true'). Use to exercise features "
            "without a dedicated flag."
        ),
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of invocations to keep in flight in parallel (default: 1)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=None,
        help="Total number of invocations to submit (default: same as --concurrency)",
    )
    parser.add_argument(
        "--invocation-timeout",
        type=int,
        default=DEFAULT_INVOCATION_TIMEOUT_S,
        metavar="SECONDS",
        help=(
            f"Per-invocation timeout passed to SageMaker (default: "
            f"{DEFAULT_INVOCATION_TIMEOUT_S}; max: {ASYNC_INVOKE_MAX_TIMEOUT_S})."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_S,
        metavar="SECONDS",
        help=f"Seconds between S3 polls for each invocation's result (default: {DEFAULT_POLL_INTERVAL_S}).",
    )
    parser.add_argument(
        "--inference-id-prefix",
        default=None,
        metavar="PREFIX",
        help=(
            "Override the InferenceId prefix sent to SageMaker (default: a fresh "
            "UUID per run). Useful for correlating multiple invocations in logs."
        ),
    )
    parser.add_argument(
        "--show-transcript",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print the transcript head of each successful invocation (default: enabled).",
    )
    parser.add_argument(
        "--transcript-chars",
        type=int,
        default=300,
        help="Characters of each transcript to print (default: 300).",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO)",
    )
    return parser


def main() -> int:
    parser = _make_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.concurrency < 1:
        print("ERROR: --concurrency must be a positive integer (minimum 1)")
        return 1
    if args.requests is None:
        args.requests = args.concurrency
    if args.requests < 1:
        print("ERROR: --requests must be a positive integer (minimum 1)")
        return 1
    if args.concurrency > args.requests:
        print("WARNING: --concurrency exceeds --requests; clamping to request count")
        args.concurrency = args.requests
    if args.invocation_timeout < 1 or args.invocation_timeout > ASYNC_INVOKE_MAX_TIMEOUT_S:
        print(
            f"ERROR: --invocation-timeout must be between 1 and "
            f"{ASYNC_INVOKE_MAX_TIMEOUT_S} seconds"
        )
        return 1

    args.redact = _parse_redact(args.redact)
    args.keyterms = [kt.strip() for kt in args.keyterms.split(",") if kt.strip()]
    extra_pairs = _parse_extra(args.extra)
    custom_attributes = _build_async_query_string(args, extra_pairs)

    # WAV validation (always — even when --input-s3-uri is supplied, the local
    # file is the source of truth for sample-rate / duration printing).
    try:
        sample_rate, channels, duration_s, size_bytes = _validate_wav(args.file)
    except (FileNotFoundError, ValueError, wave.Error) as e:
        print(f"ERROR: {e}")
        return 1

    client = AsyncSTTClient(
        endpoint_name=args.endpoint_name,
        region=args.region,
        poll_interval_s=args.poll_interval,
        invocation_timeout_s=args.invocation_timeout,
    )

    # Resolve the input S3 URI: upload-from-disk or use-as-is.
    if args.input_s3_uri:
        input_s3_uri = args.input_s3_uri
        upload_action = f"using existing object {input_s3_uri}"
    else:
        key = f"{args.upload_prefix.strip('/')}/{uuid.uuid4()}.wav"
        try:
            input_s3_uri = client.upload_wav(args.file, args.bucket, key)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            msg = e.response.get("Error", {}).get("Message", str(e))
            print(f"ERROR: S3 upload failed [{code}]: {msg}")
            return 1
        upload_action = f"uploaded to {input_s3_uri}"

    inference_id_prefix = args.inference_id_prefix or f"async-{uuid.uuid4().hex[:8]}"

    print("=" * 60)
    print("Deepgram SageMaker WAV Async Client (S3 In/Out)")
    print("=" * 60)
    print(f"Endpoint:        {args.endpoint_name}")
    print(f"WAV File:        {args.file}")
    print(f"File Size:       {size_bytes / (1024 * 1024):.1f} MiB")
    print(f"Duration:        {duration_s:.2f}s")
    print(f"Sample Rate:     {sample_rate} Hz")
    print(f"Channels:        {channels}")
    print(f"Model:           {args.model}")
    print(f"Language:        {args.language}")
    print(f"Region:          {args.region}")
    print(f"Input:           {upload_action}")
    print(f"Requests:        {args.requests}")
    print(f"Concurrency:     {args.concurrency}")
    print(f"Per-inv timeout: {args.invocation_timeout}s")
    print(f"Poll interval:   {args.poll_interval}s")
    print(f"InferenceId:     {inference_id_prefix}-NNN")
    print(f"Path+Params:     {custom_attributes}")
    if args.redact:
        print(f"Redact:          {', '.join(args.redact)}")
    if args.keyterms:
        print(f"Keyterms:        {', '.join(args.keyterms)}")
    if extra_pairs:
        print(f"Extra:           {'&'.join(f'{k}={v}' for k, v in extra_pairs)}")
    print("=" * 60)
    print(f"\nSubmitting {args.requests} request(s) ({args.concurrency} concurrent)...\n")

    wall_start = time.monotonic()
    results = client.run(
        input_s3_uri=input_s3_uri,
        custom_attributes=custom_attributes,
        inference_id_prefix=inference_id_prefix,
        num_requests=args.requests,
        concurrency=args.concurrency,
    )
    wall_elapsed = time.monotonic() - wall_start

    results.sort(key=lambda r: r[0])

    successes = 0
    failures = 0
    latencies: list[float] = []

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    for req_id, elapsed, result, output_uri, error in results:
        if error is not None:
            failures += 1
            print(f"[Request {req_id:>3}] ERROR ({elapsed:.2f}s): {error}")
            if output_uri:
                print(f"             output uri: {output_uri}")
        else:
            successes += 1
            latencies.append(elapsed)
            transcript, confidence, dur = _extract_transcript(result or {})
            tail = transcript.strip()
            head_msg = (
                f"[Request {req_id:>3}] ✓ ({elapsed:.2f}s) duration={dur:.2f}s "
                f"conf={confidence:.1%} output={output_uri}"
            )
            print(head_msg)
            if args.show_transcript and tail:
                shown = tail[: args.transcript_chars]
                ellipsis = "…" if len(tail) > args.transcript_chars else ""
                print(f"             {shown}{ellipsis}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total requests:  {args.requests}")
    print(f"Successful:      {successes}")
    print(f"Failed:          {failures}")
    if latencies:
        latencies_sorted = sorted(latencies)
        p95_idx = max(0, int(len(latencies_sorted) * 0.95) - 1)
        print(f"Min latency:     {min(latencies):.2f}s")
        print(f"Avg latency:     {statistics.mean(latencies):.2f}s")
        print(f"P95 latency:     {latencies_sorted[p95_idx]:.2f}s")
        print(f"Max latency:     {max(latencies):.2f}s")
        if len(latencies) > 1:
            print(f"Std dev:         {statistics.stdev(latencies):.2f}s")
    print(f"Total wall time: {wall_elapsed:.2f}s")
    if args.requests > 0 and wall_elapsed > 0:
        print(f"Throughput:      {args.requests / wall_elapsed:.2f} req/s")
    print("=" * 60)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
