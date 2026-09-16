#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["boto3>=1.43.49", "aws-sdk-sagemaker-runtime-http2[awscrt]>=0.11"]
# ///
"""Send one real request to a Deepgram SageMaker endpoint and judge the answer.

Picks the right API path, transport and required parameters for the product,
so a passing run proves the endpoint works end to end and a failing run says
which of the classic mismatches it hit. Also prints the exact request shape so
it can be copied into the customer's application.

  Modes (--mode; default = what the endpoint/product is built for):
    streaming  InvokeEndpointWithBidirectionalStream over HTTP/2 on PORT 8443.
               STT: sends the sample WAV as linear16 chunks, expects final transcripts
               (v1/listen: Results with is_final; v2/listen Flux: TurnInfo EndOfTurn).
               TTS: sends Speak + Flush, expects non-silent audio bytes.
    sync       InvokeEndpoint with the Deepgram query in CustomAttributes
               ("v1/listen?model=nova-3&language=en"). Real-time endpoints only.
    async      InvokeEndpointAsync against an EXISTING asynchronous endpoint (diagnosis only —
               asynchronous endpoints are temporarily not supported for Marketplace-hosted
               Deepgram; contact a Deepgram representative). Needs --async-bucket.

  uv run invoke_test.py my-endpoint --region us-east-2 --product nova-3-mono-streaming
  uv run invoke_test.py my-endpoint --region us-east-2 --product nova-3-mono-batch --mode sync
  uv run invoke_test.py my-endpoint --region us-east-2 --product aura-2 --text "Hello from SageMaker"
  uv run invoke_test.py my-endpoint --region us-east-2 --product nova-3-mono-streaming --language es --audio my.wav

Exit 0 pass · 1 the request was answered but the result is not acceptable (or a
known mismatch was detected) · 2 AWS/transport error.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import wave
from urllib.parse import quote

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from _common import (ASSETS_DIR, EXIT_NEGATIVE, EXIT_OK, add_common_args, aws_error_text, client,
                     fail_error, find_product, finish, load_catalog, make_session, parse_kv_list,
                     product_choices, resolve_region, say)

DEFAULT_AUDIO = ASSETS_DIR / "spacewalk.wav"
DEFAULT_TEXT = "Hello from Deepgram on Amazon SageMaker. If you can hear this, the endpoint is working."
TTS_SAMPLE_RATE = 24000
PRE_UPGRADE_REJECT = (
    "SageMaker refused to open the WebSocket (HTTP 424 'Failed to establish WebSocket connection'): the "
    "container rejected the request BEFORE the stream was established. SageMaker does not forward the "
    "container's own error message to streaming clients, so the reason has to be inferred: most often the "
    "model/language does not match the deployed version (multilingual listings need language=multi; Flux "
    "multi needs model=flux-general-multi and no language=), or the API path is wrong for the product. The "
    "container log (endpoint_status.py) has the exact 400."
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def read_wav(path: str) -> tuple[bytes, int, int]:
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            fail_error(f"{path}: need 16-bit PCM WAV (got {w.getsampwidth() * 8}-bit)")
        return w.readframes(w.getnframes()), w.getframerate(), w.getnchannels()


def qs(params: dict) -> str:
    return "&".join(f"{k}={quote(str(v), safe='')}" for k, v in params.items())


def explain_400(body: str, prod: dict | None, mode: str) -> str | None:
    b = body.lower()
    if "no such model" in b or "no model matched" in b:
        if prod and prod["transport"] == "streaming" and mode == "sync":
            return ("This is a STREAMING listing: synchronous InvokeEndpoint is rejected by design. "
                    "Use --mode streaming, or deploy the Batch listing for HTTP requests.")
        if prod and prod.get("languages") == "multilingual":
            return "Multilingual listing: the request must use language=multi."
        return ("The model/language in the request is not in the deployed version. Check the version's "
                "language list (resolve_model_package_arn.py) and pass --language accordingly.")
    if "unsupported_parameter" in b or "unknown" in b and "param" in b:
        return "A query parameter is not accepted by this product (unknown parameters are rejected, not ignored)."
    if "404" in b or "not found" in b:
        return "The API path is missing or wrong: CustomAttributes must start with v1/listen, v2/listen, v1/speak or v2/speak."
    return None


def audio_looks_real(pcm: bytes) -> bool:
    if len(pcm) < 4000:
        return False
    import array
    samples = array.array("h", pcm[: len(pcm) - (len(pcm) % 2)])
    peak = max(abs(s) for s in samples[::4]) if samples else 0
    return peak > 200


# ---------------------------------------------------------------------------
# sync + async
# ---------------------------------------------------------------------------

def run_sync(rt, name: str, prod: dict, params: dict, audio_path: str, text: str, result: dict) -> int:
    attrs = f"{prod['api_path']}?{qs(params)}"
    if prod["service"] == "stt":
        read_wav(audio_path)  # validates the format
        content_type = "audio/wav"
        with open(audio_path, "rb") as f:
            body = f.read()
    else:
        body = json.dumps({"text": text}).encode()
        content_type = "application/json"
    result["request"] = {"api": "InvokeEndpoint", "EndpointName": name, "ContentType": content_type,
                         "Accept": "application/json" if prod["service"] == "stt" else "*/*",
                         "CustomAttributes": attrs, "body": f"<{len(body)} bytes>"}
    say(f"InvokeEndpoint  CustomAttributes={attrs!r}  ContentType={content_type}  body={len(body)} bytes")
    t0 = time.monotonic()
    try:
        resp = rt.invoke_endpoint(EndpointName=name, ContentType=content_type,
                                  CustomAttributes=attrs, Body=body,
                                  **({"Accept": "application/json"} if prod["service"] == "stt" else {}))
    except ClientError as e:
        err = e.response.get("Error", {})
        msg = err.get("Message", "")
        # ModelError carries the container's HTTP status + body
        status = e.response.get("OriginalStatusCode")
        result.update(error=aws_error_text(e), container_status=status, container_message=msg[:500])
        why = explain_400(msg, prod, "sync")
        if why:
            result["diagnosis"] = why
            say(f"endpoint answered {status or err.get('Code')}: {msg[:300]}")
            say(f"→ {why}")
            return EXIT_NEGATIVE
        fail_error("InvokeEndpoint failed", e)
    except BotoCoreError as e:
        fail_error("InvokeEndpoint failed", e)
    latency = time.monotonic() - t0
    payload = resp["Body"].read()
    result["latency_s"] = round(latency, 2)
    if prod["service"] == "stt":
        try:
            j = json.loads(payload)
        except json.JSONDecodeError:
            result["raw"] = payload[:300].decode(errors="replace")
            say("response was not JSON")
            return EXIT_NEGATIVE
        alt = ((j.get("results") or {}).get("channels") or [{}])[0].get("alternatives") or [{}]
        transcript = alt[0].get("transcript", "")
        result.update(transcript=transcript, request_id=(j.get("metadata") or {}).get("request_id"),
                      duration_s=(j.get("metadata") or {}).get("duration"))
        say(f"transcript ({latency:.1f}s): {transcript[:200]!r}")
        return EXIT_OK if transcript.strip() else EXIT_NEGATIVE
    ct = resp.get("ContentType", "")
    ok = audio_looks_real(payload) if params.get("encoding") == "linear16" else len(payload) > 1000
    result.update(audio_bytes=len(payload), content_type=ct, audio_ok=ok)
    say(f"audio: {len(payload)} bytes, {ct} ({latency:.1f}s) → {'ok' if ok else 'EMPTY/SILENT'}")
    return EXIT_OK if ok else EXIT_NEGATIVE


def run_async(session, rt, region: str, fips: bool, name: str, prod: dict, params: dict,
              audio_path: str, bucket: str, wait_minutes: int, result: dict) -> int:
    s3 = client(session, "s3", region=region, fips=fips)
    key = f"deepgram-invoke-test/{name}/{int(time.time())}/{os.path.basename(audio_path)}"
    try:
        s3.upload_file(audio_path, bucket, key)
    except (ClientError, BotoCoreError) as e:
        fail_error(f"upload to s3://{bucket}/{key} failed (does this identity have s3:PutObject?)", e)
    attrs = f"{prod['api_path']}?{qs(params)}"
    result["request"] = {"api": "InvokeEndpointAsync", "EndpointName": name, "InputLocation": f"s3://{bucket}/{key}",
                         "ContentType": "audio/wav", "Accept": "application/json", "CustomAttributes": attrs,
                         "InvocationTimeoutSeconds": 3600}
    say(f"InvokeEndpointAsync  InputLocation=s3://{bucket}/{key}  CustomAttributes={attrs!r}")
    try:
        resp = rt.invoke_endpoint_async(EndpointName=name, InputLocation=f"s3://{bucket}/{key}",
                                        ContentType="audio/wav", Accept="application/json",
                                        CustomAttributes=attrs, InvocationTimeoutSeconds=3600)
    except ClientError as e:
        msg = e.response.get("Error", {}).get("Message", "")
        if "not an async" in msg.lower() or "AsyncInferenceConfig" in msg or e.response.get("Error", {}).get("Code") == "ValidationError":
            result.update(error=aws_error_text(e),
                          diagnosis="This endpoint is not asynchronous. Deploy with --async-bucket for InvokeEndpointAsync, or use --mode sync.")
            say(result["diagnosis"])
            return EXIT_NEGATIVE
        fail_error("InvokeEndpointAsync failed", e)
    except BotoCoreError as e:
        fail_error("InvokeEndpointAsync failed", e)
    out_loc, fail_loc = resp["OutputLocation"], resp.get("FailureLocation")
    result.update(output_location=out_loc, failure_location=fail_loc, inference_id=resp.get("InferenceId"))
    say(f"queued: output → {out_loc}")

    def split(uri):
        b, _, k = uri[5:].partition("/")
        return b, k

    deadline = time.monotonic() + wait_minutes * 60
    t0 = time.monotonic()
    while time.monotonic() < deadline:
        for loc, kind in ((out_loc, "output"), (fail_loc, "failure")):
            if not loc:
                continue
            b, k = split(loc)
            try:
                body = s3.get_object(Bucket=b, Key=k)["Body"].read()
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
                    continue
                fail_error(f"reading {loc} failed", e)
            result["latency_s"] = round(time.monotonic() - t0, 1)
            if kind == "failure":
                result.update(failure_body=body[:500].decode(errors="replace"))
                why = explain_400(result["failure_body"], prod, "async")
                if why:
                    result["diagnosis"] = why
                say(f"FAILED: {result['failure_body'][:300]}" + (f"\n→ {why}" if why else ""))
                return EXIT_NEGATIVE
            try:
                j = json.loads(body)
            except json.JSONDecodeError:
                result["raw"] = body[:300].decode(errors="replace")
                return EXIT_NEGATIVE
            alt = ((j.get("results") or {}).get("channels") or [{}])[0].get("alternatives") or [{}]
            transcript = alt[0].get("transcript", "")
            result.update(transcript=transcript, request_id=(j.get("metadata") or {}).get("request_id"))
            say(f"transcript ({result['latency_s']}s): {transcript[:200]!r}")
            return EXIT_OK if transcript.strip() else EXIT_NEGATIVE
        time.sleep(5)
    say(f"no output after {wait_minutes} min (a scaled-to-zero async endpoint can take several minutes to start)")
    result["diagnosis"] = "timed out waiting for the async result"
    return EXIT_NEGATIVE


# ---------------------------------------------------------------------------
# streaming (HTTP/2 bidirectional, port 8443)
# ---------------------------------------------------------------------------

def bidi_endpoint_uri(region: str, fips: bool) -> str:
    """runtime[-fips].sagemaker.<region>.amazonaws.com:8443 — derived from botocore's own
    resolution so --fips moves the stream too. The port is mandatory: on 443 the
    connection is accepted and never answered."""
    from botocore.config import Config as BotoConfig
    url = boto3.Session(region_name=region).client(
        "sagemaker-runtime", config=BotoConfig(use_fips_endpoint=True) if fips else None).meta.endpoint_url
    return f"{url}:8443"


async def make_bidi_client(session, region: str, fips: bool):
    """HTTP/2 SageMaker runtime client (aws-sdk-sagemaker-runtime-http2 >= 0.11).

    The config must be built with `await ...Config.resolve(...)`. Credentials are
    the ones the boto3 chain resolved (profile / SSO / role), passed as static
    overrides so AWS_PROFILE and SSO work without exporting env vars. Transport
    is the AWS CRT client (the aiohttp default does not do the HTTP/2
    bidirectional stream). The endpoint URI carries the mandatory :8443.
    """
    from aws_sdk_sagemaker_runtime_http2.client import AsyncSageMakerRuntimeHTTP2Client
    from aws_sdk_sagemaker_runtime_http2.config import AsyncSageMakerRuntimeHTTP2Config
    from smithy_http.aio.crt import AWSCRTHTTPClient

    creds = session.get_credentials()
    if creds is None:
        fail_error("no AWS credentials found (set AWS_PROFILE or AWS_ACCESS_KEY_ID/…)")
    frozen = creds.get_frozen_credentials()
    overrides = dict(
        region=region,
        endpoint_uri=bidi_endpoint_uri(region, fips),
        aws_access_key_id=frozen.access_key,
        aws_secret_access_key=frozen.secret_key,
        transport=AWSCRTHTTPClient(),
    )
    if frozen.token:
        overrides["aws_session_token"] = frozen.token
    cfg = await AsyncSageMakerRuntimeHTTP2Config.resolve(**overrides)
    return AsyncSageMakerRuntimeHTTP2Client(config=cfg)


async def stream_once(session, name: str, region: str, fips: bool, prod: dict, params: dict,
                      audio_path: str, text: str, result: dict, pace: float, max_seconds: float) -> int:
    from aws_sdk_sagemaker_runtime_http2.models import (InvokeEndpointWithBidirectionalStreamInput,
                                                        RequestPayloadPart, RequestStreamEventPayloadPart,
                                                        ResponseStreamEventPayloadPart)

    uri = bidi_endpoint_uri(region, fips)
    is_stt = prod["service"] == "stt"
    if is_stt:
        pcm, sr, ch = read_wav(audio_path)
        params = {**params, "encoding": "linear16", "sample_rate": sr}
        if ch > 1:
            params["channels"] = ch
    else:
        params = {"encoding": "linear16", "sample_rate": TTS_SAMPLE_RATE, **params}
    query = qs(params)
    result["request"] = {"api": "InvokeEndpointWithBidirectionalStream", "endpoint_uri": uri,
                         "EndpointName": name, "ModelInvocationPath": prod["api_path"],
                         "ModelQueryString": query}
    say(f"bidi stream  {uri}  path={prod['api_path']}  query={query}")

    cli = await make_bidi_client(session, region, fips)
    stream = await asyncio.wait_for(cli.invoke_endpoint_with_bidirectional_stream(
        InvokeEndpointWithBidirectionalStreamInput(endpoint_name=name, model_invocation_path=prod["api_path"],
                                                   model_query_string=query)), timeout=30)
    output = await asyncio.wait_for(stream.await_output(), timeout=30)
    out_stream = output[1]

    async def send(b: bytes, data_type: str):
        await stream.input_stream.send(RequestStreamEventPayloadPart(
            value=RequestPayloadPart(bytes_=b, data_type=data_type)))

    finals: list[str] = []
    interims = 0
    audio = bytearray()
    messages: dict[str, int] = {}
    errors: list[str] = []
    done = asyncio.Event()

    async def reader():
        nonlocal interims
        try:
            while True:
                r = await out_stream.receive()
                if r is None:
                    break
                if not isinstance(r, ResponseStreamEventPayloadPart):
                    # Typed error events (ModelStreamError / InternalStreamFailure) — the
                    # 0.11 client surfaces server-side rejections here instead of hanging.
                    val = getattr(r, "value", None)
                    detail = f"{type(r).__name__}: {getattr(val, 'message', None) or val!r}"
                    code = getattr(val, "error_code", None)
                    errors.append(f"{detail}" + (f" (error_code={code})" if code else ""))
                    say(f"  ! {errors[-1][:300]}")
                    continue
                if not (r.value and r.value.bytes_):
                    continue
                raw = r.value.bytes_
                try:
                    m = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    audio.extend(raw)  # TTS binary audio frame
                    continue
                t = m.get("type") or ("Results" if "channel" in m else "?")
                messages[t] = messages.get(t, 0) + 1
                if t == "Results":
                    alt = ((m.get("channel") or {}).get("alternatives") or [{}])[0]
                    tr = alt.get("transcript", "")
                    if tr.strip():
                        if m.get("is_final"):
                            finals.append(tr)
                            say(f"  ✓ {tr}")
                        else:
                            interims += 1
                elif t == "TurnInfo":
                    ev = m.get("event")
                    if ev == "EndOfTurn" and m.get("transcript", "").strip():
                        finals.append(m["transcript"])
                        say(f"  ✓ [EndOfTurn] {m['transcript']}")
                elif t in ("Error",) or m.get("error"):
                    errors.append(json.dumps(m)[:300])
                    say(f"  ! {json.dumps(m)[:300]}")
                elif t == "SpeechMetadata":
                    say(f"  turn metadata: {json.dumps(m)[:200]}")
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "Input stream broken" not in msg and "closed" not in msg.lower():
                errors.append(f"{type(e).__name__}: {msg[:300]}")
        finally:
            done.set()

    rtask = asyncio.create_task(reader())
    try:
        if is_stt:
            chunk = int(sr * ch * 2 * 0.1)  # 100 ms
            for i in range(0, len(pcm), chunk):
                await send(pcm[i:i + chunk], "BINARY")
                await asyncio.sleep(0.1 / pace)
            await send(json.dumps({"type": "CloseStream"}).encode(), "UTF8")
        else:
            await send(json.dumps({"type": "Speak", "text": text}).encode(), "UTF8")
            await send(json.dumps({"type": "Flush"}).encode(), "UTF8")
            # give synthesis time, then close the session
            for _ in range(int(max_seconds * 2)):
                await asyncio.sleep(0.5)
                if messages.get("SpeechMetadata") or (prod["api_path"] == "v1/speak" and messages.get("Flushed") and len(audio) > 0):
                    await asyncio.sleep(1.0)
                    break
            await send(json.dumps({"type": "Close"}).encode(), "UTF8")
        try:
            await asyncio.wait_for(done.wait(), timeout=max_seconds)
        except asyncio.TimeoutError:
            say("  (server did not close the stream in time; closing from our side)")
    finally:
        try:
            await stream.input_stream.close()
        except Exception:  # noqa: BLE001
            pass
        if not done.is_set():
            try:
                await asyncio.wait_for(done.wait(), timeout=5)
            except asyncio.TimeoutError:
                rtask.cancel()
        try:
            await cli.close()
        except Exception:  # noqa: BLE001
            pass

    result.update(messages=messages, errors=errors)
    if is_stt:
        result.update(finals=finals, interims=interims, transcript=" ".join(finals))
        if not finals:
            why = None
            if errors:
                why = explain_400(" ".join(errors), prod, "streaming")
            if why is None and not messages:
                why = ("No messages at all came back. Check: the endpoint is InService, the IAM identity has "
                       "sagemaker:InvokeEndpointWithBidirectionalStream, and the region matches. A wrong "
                       "language/model closes the stream almost immediately with no transcript.")
            if why is None:
                why = ("The stream opened but produced no final transcript. Most often the language/model "
                       "does not match the deployed version (multilingual needs language=multi; Flux multi "
                       "needs model=flux-general-multi).")
            result["diagnosis"] = why
            say(f"→ {why}")
            return EXIT_NEGATIVE
        return EXIT_OK
    ok = audio_looks_real(bytes(audio))
    result.update(audio_bytes=len(audio), audio_ok=ok,
                  audio_seconds=round(len(audio) / (TTS_SAMPLE_RATE * 2), 2))
    say(f"audio: {len(audio)} bytes ≈ {result['audio_seconds']}s of {TTS_SAMPLE_RATE} Hz PCM → {'ok' if ok else 'EMPTY/SILENT'}")
    if not ok and errors:
        result["diagnosis"] = explain_400(" ".join(errors), prod, "streaming") or errors[0]
    return EXIT_OK if ok else EXIT_NEGATIVE


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="endpoint name")
    add_common_args(p)
    p.add_argument("--product", required=True, help="catalog slug the endpoint was deployed from")
    p.add_argument("--mode", choices=["streaming", "sync", "async"], default=None)
    p.add_argument("--language", default=None, help="STT language (default from the product; multilingual is fixed to multi)")
    p.add_argument("--model", default=None, help="override model / TTS voice (e.g. aura-2-thalia-en)")
    p.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="extra Deepgram query parameter")
    p.add_argument("--audio", default=str(DEFAULT_AUDIO), help="16-bit PCM WAV to send (STT)")
    p.add_argument("--text", default=DEFAULT_TEXT, help="text to synthesize (TTS)")
    p.add_argument("--async-bucket", default=None, help="S3 bucket for --mode async input")
    p.add_argument("--pace", type=float, default=2.0, help="streaming send speed relative to real time (default 2x)")
    p.add_argument("--wait-minutes", type=int, default=10, help="async: how long to wait for the result")
    p.add_argument("--timeout-s", type=float, default=90, help="streaming: max seconds to wait for the server")
    args = p.parse_args()

    catalog = load_catalog()
    prod = find_product(catalog, args.product)
    if prod is None:
        fail_error(f"unknown product {args.product!r}. Known: {product_choices(catalog)}")
    session = make_session(args)
    region = resolve_region(args, session)
    sm = client(session, "sagemaker", region=region, fips=args.fips)
    rt = client(session, "sagemaker-runtime", region=region, fips=args.fips, retries=1)

    try:
        d = sm.describe_endpoint(EndpointName=args.name)
    except (ClientError, BotoCoreError) as e:
        fail_error(f"DescribeEndpoint {args.name} in {region} failed (wrong name or region?)", e)
    if d.get("EndpointStatus") != "InService":
        say(f"endpoint is {d.get('EndpointStatus')}, not InService — run endpoint_status.py first")
        return finish(args, {"endpoint_name": args.name, "status": d.get("EndpointStatus")}, EXIT_NEGATIVE)
    is_async_ep = bool(d.get("AsyncInferenceConfig"))

    mode = args.mode
    if mode == "async" or (mode is None and is_async_ep):
        say("Asynchronous endpoints are TEMPORARILY NOT SUPPORTED for Marketplace-hosted Deepgram. "
            "Deploy a real-time endpoint and use --mode sync or streaming; for an async use case, reach out "
            "to a Deepgram representative. (--mode async still runs against an existing async endpoint for "
            "diagnosis only.)")
        if mode is None:
            mode = "async"
    if mode is None:
        if is_async_ep:
            mode = "async"
        elif "streaming" in prod["invocation_modes"]:
            mode = "streaming"
        else:
            mode = "sync"
    if is_async_ep and mode != "async":
        say("This endpoint is ASYNCHRONOUS; it only accepts InvokeEndpointAsync. Switching to --mode async.")
        mode = "async"
    if mode == "async" and not is_async_ep:
        say("This endpoint is real-time (no AsyncInferenceConfig); InvokeEndpointAsync will be rejected. Use --mode sync or streaming.")
        return finish(args, {"endpoint_name": args.name, "diagnosis": "not an async endpoint"}, EXIT_NEGATIVE)
    if mode not in prod["invocation_modes"] and not (mode == "async" and "async" in prod["invocation_modes"]):
        say(f"note: {prod['slug']} is built for {prod['invocation_modes']}; testing {mode} anyway to show the response.")

    params = dict(prod["base_params"])
    if prod["service"] == "stt" and prod["family"] != "flux":
        if args.language:
            if prod.get("languages") == "multilingual" and args.language != "multi":
                say(f"note: {prod['slug']} requires language=multi; overriding --language {args.language}")
                params["language"] = "multi"
            else:
                params["language"] = args.language
        elif prod.get("default_language"):
            params["language"] = prod["default_language"]
    if prod["family"] == "flux" and args.language:
        say("note: Flux has no language parameter (model name selects the language set); ignoring --language")
    if args.model:
        params["model"] = args.model
    params.update(parse_kv_list(args.param, "--param"))
    if mode == "sync" and prod["service"] == "tts":
        params.setdefault("encoding", "linear16")
        params.setdefault("sample_rate", TTS_SAMPLE_RATE)

    result: dict = {"endpoint_name": args.name, "region": region, "product": prod["slug"], "mode": mode,
                    "fips": args.fips}
    if mode == "sync":
        rc = run_sync(rt, args.name, prod, params, args.audio, args.text, result)
    elif mode == "async":
        if not args.async_bucket:
            fail_error("--mode async needs --async-bucket (a bucket the execution role can read/write)")
        rc = run_async(session, rt, region, args.fips, args.name, prod, params, args.audio, args.async_bucket,
                       args.wait_minutes, result)
    else:
        try:
            rc = asyncio.run(asyncio.wait_for(
                stream_once(session, args.name, region, args.fips, prod, params, args.audio, args.text, result,
                            args.pace, args.timeout_s), timeout=args.timeout_s + 60))
        except asyncio.TimeoutError:
            result["diagnosis"] = ("The stream never opened or never answered within the timeout. Check the endpoint is "
                                   "InService and the URI includes :8443; a rejected request surfaces as a 424 within "
                                   "seconds on aws-sdk-sagemaker-runtime-http2 >= 0.6 (0.4.x holds it until the input "
                                   "stream closes), so a silent hang points at the port, connectivity, or an old client.")
            say(result["diagnosis"])
            rc = EXIT_NEGATIVE
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "424" in msg or "Failed to establish WebSocket connection" in msg:
                result.update(error=f"{type(e).__name__}: {msg[:400]}", diagnosis=PRE_UPGRADE_REJECT)
                if prod.get("languages") == "multilingual" and params.get("language") != "multi":
                    result["diagnosis"] += " (This listing requires language=multi.)"
                say(f"endpoint refused the stream: {msg[:200]}")
                say(f"→ {PRE_UPGRADE_REJECT}")
                rc = EXIT_NEGATIVE
            elif "403" in msg or "AccessDenied" in msg or "not authorized" in msg:
                result.update(error=msg[:400], diagnosis="The IAM identity lacks sagemaker:InvokeEndpointWithBidirectionalStream on this endpoint.")
                say(f"→ {result['diagnosis']}")
                rc = EXIT_NEGATIVE
            else:
                fail_error("bidirectional stream failed", e)

    result["verdict"] = "PASS" if rc == EXIT_OK else "FAIL"
    say("")
    say(f"{result['verdict']}  ({mode} on {prod['slug']})")
    if rc == EXIT_OK:
        say("Request shape to reuse in your application: see 'request' in --json output.")
    return finish(args, result, rc)


if __name__ == "__main__":
    sys.exit(main())
