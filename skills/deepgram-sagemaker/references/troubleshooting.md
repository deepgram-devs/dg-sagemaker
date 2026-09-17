# Troubleshooting

Start with `endpoint_status.py <name> --region <r>`: status, FailureReason, whether
the container log group exists, last log lines and a `classification`.

## Subscribing

| Symptom | Cause | Fix |
|---|---|---|
| An AWS employee expected a Field Demonstration Program offer but `list_products.py` shows only `public` and `subscribe.py` selects the public offer | the account is not enrolled in the AWS Marketplace Field Demonstration Program; only enrolled accounts see the offer | enrollment is handled by AWS Marketplace, not by this kit — the person's AWS internal guidance covers it. Meanwhile the public offer (free trial once per product per account) or a private offer works. See `field-demonstration-program.md`. |
| `list_products.py` says a product is subscribed on the `public` offer although an FDP offer is available | the subscription predates enrollment, or was made with `--no-fdp` / in the console | only one agreement per product can be active: cancel it in the console (Manage subscriptions), then re-run `subscribe.py <slug>`; it selects the FDP offer. |
| `CreateAgreementRequest` rejected with the free-trial term | this account already used the 14-day trial for this product | `subscribe.py` retries without it automatically; or pass `--no-free-trial`. |

```bash
aws logs tail --follow /aws/sagemaker/Endpoints/<name> --region <r>
aws sagemaker describe-endpoint --endpoint-name <name> --region <r> --query '{Status:EndpointStatus,FailureReason:FailureReason}'
```

## Endpoint never reaches InService

| Signal | Cause (`classification.kind`) | Fix |
|---|---|---|
| `ResourceLimitExceeded` at create / FailureReason mentions quota | `quota` | `check_quota.py`; request an increase or use another type/region. Quota is per region. |
| `InsufficientInstanceCapacity`, or `Request to service failed.` with **no** log group | `capacity` / `container_never_started` | AWS had no hosts. `--instance-pools` with 2–3 types, or another region. Not an image problem. |
| Log says `[cuda-preflight] … set InferenceAmiVersion to al2023-ami-sagemaker-inference-gpu-4-1` | `ami_driver` | Host driver too old (console-created or unpinned config). Redeploy / `update_endpoint.py --inference-ami-version al2023-ami-sagemaker-inference-gpu-4-1`. |
| Stuck in `Creating` for a long time, then Failed with no logs | `model_download_timeout` | Large bundle. Redeploy with `--model-download-timeout-s 1800`. |
| `Caller is not subscribed to the marketplace offering` | `not_subscribed` | `subscribe.py`, or accept the private offer in each linked account (or License Manager grant). |
| `Environment variable map cannot be specified when using a ModelPackage…` | `env_not_allowed` | Listing not enabled for overrides. Remove `--env`; ask Deepgram support. |
| Instance type "not supported" | `instance_type_not_supported` | Use a type from `resolve_model_package_arn.py`'s supported list (g7/g7e only in recent versions). |
| Log group exists, container died | `container_failed` | Read the log tail; send endpoint name, region and the lines to Deepgram support. |

`/ping` returning 503 while the endpoint is `Creating` is normal (models loading);
`INFO Deepgram Engine is ready` marks the end of it.

## InService but requests fail — almost always the request, not the endpoint

| Symptom | Cause | Fix |
|---|---|---|
| 400 `No such model/language/tier` from `InvokeEndpoint` on a **Streaming** listing | Streaming listings only serve bidirectional streams | Use `InvokeEndpointWithBidirectionalStream`, or deploy the Batch listing. |
| 400 / stream closes ~1 s with no transcript on **multilingual** Nova-3 | request has `language=en` (or any specific code) | `language=multi`. |
| 400 on Flux with `language=` | Flux has no language param | `model=flux-general-en` or `model=flux-general-multi`. |
| 400 on Flux from a `/v1/listen` client | wrong path | `v2/listen` with the turn-based protocol. |
| 400 `unsupported_parameter` | unknown query param (typo, or a param this product does not carry) | remove it. |
| 404 | no API path in `CustomAttributes` / `ModelInvocationPath` | prefix with `v1/listen`, `v2/listen`, `v1/speak`, `v2/speak`. |
| Streaming client hangs, nothing arrives, `Invocations` metric stays 0 | endpoint URI lacks `:8443` | add the port. |
| Streaming open fails with **HTTP 424 `Failed to establish WebSocket connection`** (raised within seconds, input stream still open) | the container rejected the request before the upgrade (wrong model/language/path); SageMaker does not forward the container's 400 body | read the exact 400 in the container log (`endpoint_status.py`); fix `language=multi` / `model=flux-general-multi` / the path. |
| Streaming client **hangs** on open while `Invocation4XXErrors` / `InvocationModelErrors` rises | `aws-sdk-sagemaker-runtime-http2` **0.4.x** holds the 424 until the input stream is closed (measured 2026-09-16 against the same endpoint where 0.6 and 0.11 raise it in ~0.4 s) | upgrade the client (0.11 is current; the API changed — see `invoke.md`); always bound the open with a timeout; closing the input stream releases the error on old clients. |
| `AccessDeniedException` | operator lacks the invoke action | `sagemaker:InvokeEndpoint` / `InvokeEndpointWithBidirectionalStream` / `InvokeEndpointAsync`. |
| Someone asks for `InvokeEndpointAsync` / scale-to-zero | asynchronous endpoints are temporarily not supported for Marketplace-hosted Deepgram | use a real-time endpoint; reach out to a Deepgram representative for the async use case. |
| Empty / garbage transcripts on Flux, endpoint healthy | running on `ml.g4dn` (T4) | redeploy on g5/g6/g6e/g7. |
| Aura-2 fails to load models | single-GPU instance | `ml.*.12xlarge`. |

## Health and recovery

- Streaming connections have their own liveness ping; an unanswered ping closes
  **that connection only** — clients should reconnect. Connections are capped at
  30 minutes.
- SageMaker replaces an instance when `/ping` fails repeatedly; a degraded
  container reports `sagemaker_endpoint_health{state="degraded"|"critical"}` and
  `sagemaker_time_to_critical_seconds` on `/metrics` (visible with detailed observability).
- `composite health check failed` in the log is never expected — escalate with
  endpoint name, region and surrounding lines.

## Before contacting support, collect

Endpoint name and region · FailureReason · the log tail · the exact request
(path, query, headers) · `resolve_model_package_arn.py` output (version + ARN) ·
instance type and `InferenceAmiVersion` · whether the same request works on a
Batch endpoint via `InvokeEndpoint`.
