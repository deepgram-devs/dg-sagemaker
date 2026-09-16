#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["boto3>=1.43.49"]
# ///
"""Deploy a Deepgram Marketplace model package as a SageMaker endpoint (Model + EndpointConfig + Endpoint).

Everything the Deepgram docs call non-negotiable is enforced here, so the
caller only chooses capacity and mode:

  EnableNetworkIsolation = True         mandatory for Marketplace model packages
  InferenceAmiVersion    = al2023-ami-sagemaker-inference-gpu-4-1 (NVIDIA 580 / CUDA 13)
                                        — the console cannot set this; without it the
                                        container refuses to start on g4dn/g5 hosts
  VariantName            = AllTraffic   the update / Terraform paths assume it
  ModelDataDownloadTimeoutInSeconds ≥ 600 (1800 for multilingual Nova-3 bundles)
  instance type ∈ the package's SupportedRealtimeInferenceInstanceTypes (verified live)
  async: ClientConfig.MaxConcurrentInvocationsPerInstance = 32 (SageMaker's default of 4 is too low)

Capacity is an EXPLICIT choice — pass --instance-type or --instance-pools (up to
5 types, tried in order when one has no capacity). There is no default.

  uv run deploy_endpoint.py --region us-east-2 --product nova-3-mono-streaming \
      --model-package-arn arn:aws:sagemaker:us-east-2:…:model-package/… \
      --execution-role-arn arn:aws:iam::123456789012:role/deepgram-sagemaker-execution \
      --instance-type ml.g6.2xlarge --yes
  # NOTE: asynchronous endpoints (--async-bucket) are temporarily NOT supported for
  # Marketplace-hosted Deepgram; the script refuses. Contact a Deepgram representative.
  # engine/API overrides (only on listings where AWS has enabled them):
  uv run deploy_endpoint.py … --env DEEPGRAM_ENGINE_01=max_active_requests=120

Prints the plan and requires --yes (instances bill from the moment the endpoint
is Creating). Waits for InService by default; on Failed it captures
FailureReason, checks whether a container log group exists, tails the logs and
names the likely cause with a next step.

Exit 0 InService · 1 Failed / not InService in time · 2 AWS error · 3 needs --yes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time

from botocore.exceptions import BotoCoreError, ClientError

from _common import (DEFAULT_INFERENCE_AMI, EXIT_NEGATIVE, EXIT_OK, INFERENCE_AMI_DRIVERS,
                     add_common_args, aws_error_text, check_botocore, classify_endpoint_failure,
                     client, confirm_or_exit, fail_error, find_product, finish, load_catalog,
                     log_group_state, make_session, parse_kv_list, product_choices,
                     resolve_region, say, tail_endpoint_logs)

MAX_INSTANCE_POOLS = 5
DEFAULT_RUNG_TIMEOUT_S = 300
ASYNC_UNSUPPORTED = ("Asynchronous endpoints are TEMPORARILY NOT SUPPORTED for Marketplace-hosted Deepgram. Deploy a real-time endpoint (drop --async-bucket) and use InvokeEndpoint / streaming. If you have an asynchronous use case, reach out to a Deepgram representative.")
DEFAULT_ASYNC_MAX_CONCURRENCY = 32
VALID_METRIC_FREQ = (10, 30, 60, 120, 180, 240, 300)


def wait_for_endpoint(sm, logs, name: str, max_minutes: int) -> tuple[str, dict]:
    """Poll DescribeEndpoint until InService/Failed or timeout. Returns (status, diagnostics)."""
    deadline = time.monotonic() + max_minutes * 60
    last = None
    while True:
        try:
            d = sm.describe_endpoint(EndpointName=name)
        except (ClientError, BotoCoreError) as e:
            say(f"  describe_endpoint: {aws_error_text(e)} (retrying)")
            time.sleep(30)
            continue
        status = d.get("EndpointStatus")
        if status != last:
            say(f"  {dt.datetime.now().strftime('%H:%M:%S')}  {status}")
            last = status
        if status == "InService":
            return status, {}
        if status in ("Failed", "OutOfService", "RollingBack"):
            fr = d.get("FailureReason")
            exists, note = log_group_state(logs, name)
            lines = tail_endpoint_logs(logs, name, 40) if exists else []
            diag = {"failure_reason": fr, "log_group": note, "log_tail": lines,
                    "classification": classify_endpoint_failure(status, fr, exists, lines)}
            return status, diag
        if time.monotonic() > deadline:
            exists, note = log_group_state(logs, name)
            lines = tail_endpoint_logs(logs, name, 40) if exists else []
            diag = {"failure_reason": None, "log_group": note, "log_tail": lines,
                    "classification": {"kind": "still_creating",
                                       "next_step": f"Still {status} after {max_minutes} min. Large "
                                                    "bundles can take longer; keep watching with "
                                                    "endpoint_status.py --wait, or tear down and retry "
                                                    "with a longer --model-download-timeout-s."}}
            return status, diag
        time.sleep(30)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--name", default=None,
                   help="endpoint name (also used for the Model and EndpointConfig). "
                        "Default deepgram-<product>-<UTC timestamp>")
    p.add_argument("--product", default=None,
                   help="catalog slug of the listing this ARN came from (enables mode checks + defaults)")
    p.add_argument("--model-package-arn", required=True)
    p.add_argument("--execution-role-arn", required=True,
                   help="from create_execution_role.py")
    cap = p.add_mutually_exclusive_group(required=True)
    cap.add_argument("--instance-type", default=None, help="one ml.* type")
    cap.add_argument("--instance-pools", default=None,
                     help=f"RECOMMENDED. Comma-separated ml.* types in priority order (max {MAX_INSTANCE_POOLS}); "
                          "SageMaker falls back down the list when a type has no capacity. "
                          "'default' = the product's default_pool from references/products.json (needs --product). "
                          "See references/instance-pools.md")
    p.add_argument("--rung-timeout-s", type=int, default=DEFAULT_RUNG_TIMEOUT_S,
                   help=f"VariantInstanceProvisionTimeoutInSeconds: how long SageMaker tries one pool rung "
                        f"before falling back to the next (default {DEFAULT_RUNG_TIMEOUT_S}; AWS accepts 60–3600). "
                        "Pools only.")
    p.add_argument("--single-type-reason", default=None,
                   help="why a single --instance-type is used instead of a pool (Savings Plan, measured "
                        "concurrency target, reproducing a type-specific issue). Recorded in the output.")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument("--inference-ami-version", default=DEFAULT_INFERENCE_AMI,
                   help=f"host AMI (driver) pin; default {DEFAULT_INFERENCE_AMI}. 'none' = unpinned (not recommended)")
    p.add_argument("--model-download-timeout-s", type=int, default=None,
                   help="ModelDataDownloadTimeoutInSeconds (default 600; 1800 for multilingual Nova-3)")
    p.add_argument("--startup-timeout-s", type=int, default=900,
                   help="ContainerStartupHealthCheckTimeoutInSeconds (default 900)")
    p.add_argument("--async-bucket", default=None,
                   help="make this an ASYNCHRONOUS endpoint (TEMPORARILY NOT SUPPORTED for Marketplace-hosted "
                        "Deepgram — the script refuses unless --allow-unsupported-async is also given, for "
                        "use only with a Deepgram representative)")
    p.add_argument("--allow-unsupported-async", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--async-prefix", default=None, help="S3 key prefix (default deepgram-async/<name>)")
    p.add_argument("--async-max-concurrency", type=int, default=DEFAULT_ASYNC_MAX_CONCURRENCY)
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="container Environment entry, repeatable (DEEPGRAM_API_*/DEEPGRAM_ENGINE_* overrides)")
    obs = p.add_mutually_exclusive_group()
    obs.add_argument("--detailed-observability", dest="detailed_observability", action="store_true", default=None,
                     help="MetricsConfig.EnableDetailedObservability (Prometheus/OTel metrics into CloudWatch). "
                          "Default: on when botocore supports it")
    obs.add_argument("--no-detailed-observability", dest="detailed_observability", action="store_false")
    p.add_argument("--metrics-frequency-s", type=int, default=60, choices=VALID_METRIC_FREQ)
    p.add_argument("--no-wait", action="store_true", help="return right after CreateEndpoint")
    p.add_argument("--wait-max-minutes", type=int, default=30)
    p.add_argument("--yes", action="store_true", help="create without prompting")
    args = p.parse_args()

    catalog = load_catalog()
    prod = find_product(catalog, args.product) if args.product else None
    if args.product and prod is None:
        fail_error(f"unknown product {args.product!r}. Known: {product_choices(catalog)}")

    if args.async_bucket and not args.allow_unsupported_async:
        say(ASYNC_UNSUPPORTED)
        return finish(args, {"error": "async_not_supported", "message": ASYNC_UNSUPPORTED}, EXIT_NEGATIVE)

    arn = args.model_package_arn.strip()
    if ":model-package/" not in arn:
        fail_error(f"--model-package-arn does not look like a ModelPackage ARN: {arn!r}")
    environment = parse_kv_list(args.env, "--env")
    env_api = [k for k in environment if k.startswith("DEEPGRAM_API_")]
    env_eng = [k for k in environment if k.startswith("DEEPGRAM_ENGINE_")]
    if len(env_api) > 8 or len(env_eng) > 8:
        fail_error("at most 8 DEEPGRAM_API_* and 8 DEEPGRAM_ENGINE_* entries are honoured")

    session = make_session(args)
    region = resolve_region(args, session)
    arn_region = arn.split(":")[3]
    if arn_region != region:
        fail_error(f"the ModelPackage ARN is in {arn_region} but --region is {region}. ARNs are per-region; "
                   "re-run resolve_model_package_arn.py for this region.")
    sm = client(session, "sagemaker", region=region, fips=args.fips)
    logs = client(session, "logs", region=region, fips=args.fips)

    use_default_pool = False
    if args.instance_type:
        types = [args.instance_type]
    elif args.instance_pools.strip().lower() in ("default", "recommended"):
        if not prod:
            fail_error("--instance-pools default needs --product so the default_pool can be looked up")
        types = list(catalog["instance_profiles"][prod["instance_profile"]]["default_pool"])
        use_default_pool = True
    else:
        types = [t.strip() for t in args.instance_pools.split(",") if t.strip()]
    types = list(dict.fromkeys(types))
    if not types:
        fail_error("no instance type given")
    if len(types) > MAX_INSTANCE_POOLS:
        fail_error(f"--instance-pools accepts at most {MAX_INSTANCE_POOLS} types")
    if not 60 <= args.rung_timeout_s <= 3600:
        fail_error("--rung-timeout-s must be between 60 and 3600")

    # live validation against the package ---------------------------------
    supported: list[str] = []
    try:
        d = sm.describe_model_package(ModelPackageName=arn)
        supported = (d.get("InferenceSpecification") or {}).get("SupportedRealtimeInferenceInstanceTypes", [])
    except (ClientError, BotoCoreError) as e:
        say(f"note: DescribeModelPackage failed ({aws_error_text(e)}); cannot pre-validate instance types")
    if supported:
        bad = [t for t in types if t not in supported]
        if bad and use_default_pool:
            # The default pool is a kit-wide suggestion; this package version may predate a type
            # (g7/g7e appear only in recent versions). Drop rather than fail.
            say(f"default pool: dropping {', '.join(bad)} — not in this package version's supported list")
            types = [t for t in types if t not in bad]
            bad = []
        if bad:
            say(f"instance type(s) not supported by this model package: {', '.join(bad)}")
            say(f"supported: {', '.join(supported)}")
            return finish(args, {"error": "unsupported_instance_type", "unsupported": bad,
                                 "supported": supported}, EXIT_NEGATIVE)
        if not types:
            fail_error("no instance type left after filtering against the package's supported list")

    # quota: pools fall back on CAPACITY, never on QUOTA. One zero-quota rung fails the whole
    # CreateEndpoint, so drop such rungs up front. Fails open if the lookup fails.
    quota_notes: list[str] = []
    try:
        sq = client(session, "service-quotas", region=region, fips=args.fips)
        quota_by_type: dict[str, float] = {}
        for page in sq.get_paginator("list_service_quotas").paginate(ServiceCode="sagemaker"):
            for q in page.get("Quotas", []):
                nm = q.get("QuotaName", "")
                if nm.endswith(" for endpoint usage"):
                    quota_by_type[nm[: -len(" for endpoint usage")]] = float(q.get("Value", 0))
        # current usage per type from the quota's own CloudWatch usage metric (5-min resolution,
        # may lag). Quota that is fully in use fails CreateEndpoint exactly like quota 0.
        cw = client(session, "cloudwatch", region=region, fips=args.fips)
        usage_by_type: dict[str, float] = {}
        for page in sq.get_paginator("list_service_quotas").paginate(ServiceCode="sagemaker"):
            for q in page.get("Quotas", []):
                nm = q.get("QuotaName", "")
                t = nm[: -len(" for endpoint usage")] if nm.endswith(" for endpoint usage") else None
                um = q.get("UsageMetric")
                if t in types and um:
                    try:
                        end = dt.datetime.now(dt.timezone.utc)
                        stat = um.get("MetricStatisticRecommendation") or "Maximum"
                        r = cw.get_metric_statistics(
                            Namespace=um["MetricNamespace"], MetricName=um["MetricName"],
                            Dimensions=[{"Name": k, "Value": v} for k, v in (um.get("MetricDimensions") or {}).items()],
                            StartTime=end - dt.timedelta(minutes=15), EndTime=end, Period=300, Statistics=[stat])
                        pts = r.get("Datapoints", [])
                        usage_by_type[t] = float(max(pts, key=lambda d: d["Timestamp"])[stat]) if pts else 0.0
                    except (ClientError, BotoCoreError):
                        pass
        zero = [t for t in types if quota_by_type.get(t) == 0]
        full = [t for t in types if t not in zero and quota_by_type.get(t) is not None
                and usage_by_type.get(t) is not None
                and usage_by_type[t] + args.instance_count > quota_by_type[t]]
        blocked = zero + full
        if blocked and len(types) > len(blocked):
            for t in zero:
                quota_notes.append(f"dropped {t} from the pool: endpoint quota is 0 in {region} "
                                   f"(check_quota.py --request N to raise it)")
            for t in full:
                quota_notes.append(f"dropped {t} from the pool: quota {quota_by_type[t]:g} is fully in use "
                                   f"({usage_by_type[t]:g} running) in {region}")
            types = [t for t in types if t not in blocked]
        elif blocked:
            quota_notes.append(f"WARNING: every requested type ({', '.join(blocked)}) has no free quota in "
                               f"{region} (quota 0 or fully in use); CreateEndpoint will fail with "
                               "ResourceLimitExceeded. Run check_quota.py --request N, free an endpoint, or "
                               "pick another type/region.")
        if not types:
            say("every type in the pool has quota 0 in this region — nothing to deploy on. "
                "Run check_quota.py --request N or choose another region.")
            return finish(args, {"error": "no_quota", "notes": quota_notes}, EXIT_NEGATIVE)
    except (ClientError, BotoCoreError) as e:
        quota_notes.append(f"note: could not read endpoint quotas ({aws_error_text(e)}); using the pool as given")
    for n in quota_notes:
        say("  !! " + n)

    if len(types) == 1 and not args.single_type_reason:
        pool_hint = ", ".join(catalog["instance_profiles"][prod["instance_profile"]]["default_pool"]) if prod else "A,B,C"
        say("NOTE: a single instance type has no capacity fallback; when AWS is short of this GPU the endpoint "
            f"fails ~3 min in with 'Request to service failed'. Recommended: --instance-pools {pool_hint} "
            "(see references/instance-pools.md). Pass --single-type-reason to record why one type is intended.")
    if prod:
        profile = catalog["instance_profiles"][prod["instance_profile"]]
        for t in types:
            for pat in profile.get("unsupported", []):
                fam = pat.split(".")[1] if pat.startswith("ml.") and "." in pat else None
                if fam and fam != "*" and t.startswith(f"ml.{fam}."):
                    say(f"WARNING: {t} is listed as unsupported for {prod['family']}: {profile['note']}")
        if args.async_bucket and "async" not in prod["invocation_modes"]:
            fail_error(f"{prod['slug']} is a {prod['transport']} listing; asynchronous endpoints only make "
                       "sense for the Batch listings (nova-3-*-batch).")

    # defaults derived from the product --------------------------------------
    download_timeout = args.model_download_timeout_s
    if download_timeout is None:
        download_timeout = 1800 if (prod and prod["family"] == "nova-3" and prod["languages"] == "multilingual") else 600
    if download_timeout < 600:
        say("note: ModelDataDownloadTimeoutInSeconds below 600 is likely to time out on Deepgram bundles")

    ok_boto, boto_ver = check_botocore()
    detailed = args.detailed_observability
    if detailed is None:
        detailed = ok_boto
    if detailed and not ok_boto:
        fail_error(f"--detailed-observability needs botocore >= 1.43.49 (have {boto_ver})")

    ami = None if args.inference_ami_version.lower() in ("none", "default", "") else args.inference_ami_version
    if ami and ami not in INFERENCE_AMI_DRIVERS:
        say(f"note: {ami} is not a GPU inference AMI this kit knows ({', '.join(INFERENCE_AMI_DRIVERS)})")
    if ami and INFERENCE_AMI_DRIVERS.get(ami, 999) < 580:
        say(f"WARNING: {ami} ships NVIDIA driver {INFERENCE_AMI_DRIVERS[ami]}; current Deepgram packages "
            f"need 580 (CUDA 13). Expect the container to refuse to start. Use {DEFAULT_INFERENCE_AMI}.")
    if ami is None:
        say("WARNING: unpinned InferenceAmiVersion — g4dn/g5 hosts default to driver 470 and the "
            "container will not start. Pass --inference-ami-version " + DEFAULT_INFERENCE_AMI)

    slug = prod["slug"] if prod else "endpoint"
    name = args.name or f"deepgram-{slug}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    async_prefix = (args.async_prefix or f"deepgram-async/{name}").strip("/")

    # plan --------------------------------------------------------------------
    variant: dict = {
        "VariantName": "AllTraffic", "ModelName": name, "InitialInstanceCount": args.instance_count,
        "ModelDataDownloadTimeoutInSeconds": download_timeout,
        "ContainerStartupHealthCheckTimeoutInSeconds": args.startup_timeout_s,
    }
    if len(types) == 1:
        variant["InstanceType"] = types[0]
    else:
        variant["InstancePools"] = [{"InstanceType": t, "Priority": i + 1} for i, t in enumerate(types)]
        variant["VariantInstanceProvisionTimeoutInSeconds"] = args.rung_timeout_s
    if ami:
        variant["InferenceAmiVersion"] = ami
    config_kwargs: dict = {"EndpointConfigName": name, "ProductionVariants": [variant]}
    if detailed:
        config_kwargs["MetricsConfig"] = {"EnableDetailedObservability": True,
                                          "MetricPublishFrequencyInSeconds": args.metrics_frequency_s}
    if args.async_bucket:
        config_kwargs["AsyncInferenceConfig"] = {
            "OutputConfig": {"S3OutputPath": f"s3://{args.async_bucket}/{async_prefix}/output",
                             "S3FailurePath": f"s3://{args.async_bucket}/{async_prefix}/failures"},
            "ClientConfig": {"MaxConcurrentInvocationsPerInstance": args.async_max_concurrency},
        }
    container: dict = {"ModelPackageName": arn}
    if environment:
        container["Environment"] = environment

    plan = {
        "endpoint_name": name, "region": region, "product": slug, "model_package_arn": arn,
        "execution_role_arn": args.execution_role_arn,
        "mode": "async" if args.async_bucket else ("streaming/sync real-time"),
        "instance_types": types, "instance_count": args.instance_count,
        "capacity_mode": "pool" if len(types) > 1 else "single_type",
        "rung_timeout_s": args.rung_timeout_s if len(types) > 1 else None,
        "single_type_reason": args.single_type_reason, "quota_notes": quota_notes,
        "inference_ami_version": ami, "network_isolation": True,
        "model_download_timeout_s": download_timeout, "startup_timeout_s": args.startup_timeout_s,
        "detailed_observability": detailed, "environment": environment,
        "async_output": config_kwargs.get("AsyncInferenceConfig", {}).get("OutputConfig"),
        "control_plane": sm.meta.endpoint_url,
    }
    say("== Deployment plan ==")
    for k, v in plan.items():
        say(f"  {k:26} {v}")
    confirm_or_exit(args, f"CREATE endpoint {name} in {region} on {' > '.join(types)} × {args.instance_count}. "
                          "GPU instance billing starts now and continues until the endpoint is deleted.")

    result: dict = {"plan": plan}
    created: list[str] = []  # for cleanup if a later create call fails
    # create ---------------------------------------------------------------------
    try:
        for attempt in range(6):
            try:
                sm.create_model(ModelName=name, ExecutionRoleArn=args.execution_role_arn,
                                PrimaryContainer=container, EnableNetworkIsolation=True)
                break
            except ClientError as e:
                msg = e.response.get("Error", {}).get("Message", "")
                # A role created moments ago is not yet assumable; propagation takes ~10 s.
                if attempt < 5 and ("Could not assume role" in msg or "is not authorized to perform: iam:PassRole" in msg
                                    or "AssumeRole" in msg):
                    say(f"  create_model: {msg.strip()[:120]} — retrying in 10 s (IAM propagation)")
                    time.sleep(10)
                    continue
                raise
        created.append("model")
        say(f"-- created Model {name}")
        sm.create_endpoint_config(**config_kwargs)
        created.append("config")
        say(f"-- created EndpointConfig {name}")
        sm.create_endpoint(EndpointName=name, EndpointConfigName=name)
        say(f"-- created Endpoint {name} (Creating)")
    except ClientError as e:
        msg = e.response.get("Error", {}).get("Message", "")
        result["error"] = aws_error_text(e)
        hint = None
        if "not subscribed" in msg.lower():
            hint = "This account is not subscribed to the product: run subscribe.py first."
        elif "Environment variable map cannot be specified" in msg:
            hint = "This listing does not allow --env overrides yet; remove --env or contact Deepgram support."
        elif "ResourceLimitExceeded" in aws_error_text(e):
            hint = "Quota: run check_quota.py --instance-types … and request an increase, or choose another type/region."
        elif "InferenceAmiVersion" in msg:
            hint = "This instance type / region may not support the requested InferenceAmiVersion."
        if hint:
            result["hint"] = hint
            say(hint)
        # No endpoint exists, so teardown_endpoint.py (which keys on the endpoint) cannot find
        # these; remove them here rather than leave an orphaned Model/EndpointConfig behind.
        for kind in reversed(created):
            try:
                if kind == "config":
                    sm.delete_endpoint_config(EndpointConfigName=name)
                    say(f"-- cleaned up EndpointConfig {name}")
                else:
                    sm.delete_model(ModelName=name)
                    say(f"-- cleaned up Model {name}")
            except (ClientError, BotoCoreError) as e2:
                say(f"-- could not clean up {kind} {name}: {aws_error_text(e2)}")
        fail_error("CreateModel/CreateEndpointConfig/CreateEndpoint failed", e)
    except BotoCoreError as e:
        fail_error("create call failed", e)

    if args.no_wait:
        result["status"] = "Creating"
        say(f"Not waiting. Check with: endpoint_status.py {name} --region {region} --wait")
        return finish(args, result, EXIT_OK)

    say(f"-- waiting for InService (up to {args.wait_max_minutes} min, polling every 30 s)")
    status, diag = wait_for_endpoint(sm, logs, name, args.wait_max_minutes)
    result["status"] = status
    result.update(diag)
    if status == "InService":
        say("")
        say(f"Endpoint {name} is InService in {region}.")
        say(f"Next: invoke_test.py {name} --region {region}" + (f" --product {slug}" if prod else ""))
        return finish(args, result, EXIT_OK)
    say("")
    say(f"Endpoint {name} did NOT reach InService: {status}")
    say(f"  FailureReason: {diag.get('failure_reason')}")
    say(f"  log group:     {diag.get('log_group')}")
    for line in diag.get("log_tail", [])[-15:]:
        say(f"    | {line[:220]}")
    c = diag["classification"]
    say(f"  cause: {c['kind']}")
    say(f"  next:  {c['next_step']}")
    say(f"The endpoint still exists (and a Failed endpoint does not bill). Remove it with: "
        f"teardown_endpoint.py {name} --region {region} --yes")
    return finish(args, result, EXIT_NEGATIVE)


if __name__ == "__main__":
    sys.exit(main())
