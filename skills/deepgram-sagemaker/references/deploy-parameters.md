# Deployment parameters

What `deploy_endpoint.py` sets, why, and what the person may change.

## Fixed (not flags)

| Setting | Value | Why |
|---|---|---|
| `EnableNetworkIsolation` | `true` | Mandatory for Marketplace model packages; SageMaker rejects the Model otherwise. The container has no outbound network. |
| `VariantName` | `AllTraffic` | `update_endpoint.py`, the docs' Terraform module and the update procedure assume it. |
| Model source | `PrimaryContainer.ModelPackageName = <ARN>` | Marketplace packages are referenced by ARN, never by image. |

## Chosen with the person

| Flag | Default | Notes |
|---|---|---|
| `--instance-pools default` / `A,B,C` (recommended) **or** `--instance-type T --single-type-reason "…"` | none — required | Pool = up to 5 types in priority order; SageMaker falls back on *capacity*. `default` = the product's `default_pool`, filtered to the package's supported types. Zero-*quota* rungs are dropped automatically (quota does not fall back). Single type only for a stated reason. See `instance-pools.md`. |
| `--instance-count` | `1` | Raise for steady load, or use auto-scaling. |
| `--rung-timeout-s` | `300` | Seconds SageMaker spends on one pool rung before falling back to the next (`VariantInstanceProvisionTimeoutInSeconds`, 60–3600). Pools only. |
| `--async-bucket B` | refused | Asynchronous endpoints are temporarily not supported for Marketplace-hosted Deepgram; the script exits 1 with the message to contact a Deepgram representative. |
| `--env KEY=VALUE` | none | Container overrides, see below. Only on listings where AWS has enabled it; otherwise `ValidationException: Environment variable map cannot be specified…` → remove and ask Deepgram support. |
| `--name` | `deepgram-<slug>-<UTC timestamp>` | Also the Model and EndpointConfig name. |

## Defaults that are safe to leave

| Flag | Default | Notes |
|---|---|---|
| `--inference-ami-version` | `al2023-ami-sagemaker-inference-gpu-4-1` | NVIDIA driver 580 / CUDA 13, required by current packages. Unpinned, g4dn/g5 hosts boot driver 470 and the container refuses to start. **The SageMaker console cannot set this field**, which is why console-created endpoints fail. |
| `--model-download-timeout-s` | `600`; `1800` for `nova-3-multi-*` | `ModelDataDownloadTimeoutInSeconds` is a ceiling, not a delay. Too low → stuck in Creating then Failed with no logs. |
| `--startup-timeout-s` | `900` | `ContainerStartupHealthCheckTimeoutInSeconds`. `/ping` returns 503 while models load; this bounds how long SageMaker waits for 200. |
| `--detailed-observability` | on (botocore ≥ 1.43.49) | `MetricsConfig.EnableDetailedObservability` — DCGM GPU, host and Deepgram container Prometheus metrics into CloudWatch's OTel store, 60 s. Off with `--no-detailed-observability`. |

### Inference AMI versions

| AMI | Driver | CUDA |
|---|---|---|
| `al2-ami-sagemaker-inference-gpu-2` | 535 | 12.2 |
| `al2-ami-sagemaker-inference-gpu-2-1` | 535 | 12.2 |
| `al2-ami-sagemaker-inference-gpu-3-1` | 550 | 12.4 |
| **`al2023-ami-sagemaker-inference-gpu-4-1`** | **580** | **13.0** |

Changing the AMI on a live endpoint (`update_endpoint.py --inference-ami-version`)
is a blue/green fleet replacement, not an in-place upgrade.

## Container environment overrides (`--env`)

`DEEPGRAM_API_<nn>` and `DEEPGRAM_ENGINE_<nn>` (nn = 01…08 each) carry one
dotted TOML key each, applied in suffix order:

```
--env DEEPGRAM_ENGINE_01=max_active_requests=120        # cap concurrent requests per instance
--env DEEPGRAM_API_01=features.entity_detection=false
--env DEEPGRAM_ENGINE_02=chunking.streaming.step=0.5
--env DEEPGRAM_API_02=emf.enabled=false                  # opt out of the Deepgram/SelfHosted usage metrics
```
Quote string values *inside* the expression (`key="value"`); do not quote
booleans or numbers. Most deployments need none of these. Verify in CloudWatch:
`INFO Applying to engine.toml: …` / `Successfully updated engine.toml`.

## Endpoint type and limits

Marketplace-hosted Deepgram endpoints are **real-time** endpoints: they accept
`InvokeEndpoint` (synchronous, ≤ 25 MB per request) and
`InvokeEndpointWithBidirectionalStream` (streaming, ≤ 30 min per connection),
scale from a minimum of 1 instance on `ConcurrentRequestsPerModel`.
**Asynchronous endpoints** (`InvokeEndpointAsync`, 1 GB S3 inputs, scale-to-zero)
are temporarily not supported; customers with that use case should reach out to
a Deepgram representative.

## What the deploy prints on failure

`FailureReason` (captured before anything is deleted), whether
`/aws/sagemaker/Endpoints/<name>` exists (absent ⇒ container never started ⇒
provisioning-level), the last log lines, and a `classification.kind`:
`quota`, `capacity`, `ami_driver`, `not_subscribed`, `env_not_allowed`,
`instance_type_not_supported`, `container_never_started`,
`model_download_timeout`, `container_failed`, `still_creating`. Each comes with a
`next_step`. See `troubleshooting.md`.
