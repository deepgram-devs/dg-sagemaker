#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["boto3>=1.43.49"]
# ///
"""Update a live Deepgram SageMaker endpoint in place (new instance type/count, AMI, env, model version).

Creates a NEW EndpointConfig (and a new Model when the model package ARN or the
Environment changes) by copying the current one with your overrides, then calls
UpdateEndpoint. The endpoint name and its clients do not change.

How SageMaker applies it for Marketplace containers: blue/green, all-at-once.
A full new fleet is provisioned, traffic cuts over in one step, the old fleet is
terminated. No canary, no automatic rollback. Consequences:
  * you need GPU quota for BOTH fleets for a few minutes
  * live streaming connections finish on the old instances; update during low traffic
  * roll back manually with --rollback-to <previous config name> (printed by this script)

  uv run update_endpoint.py my-endpoint --region us-east-2 --instance-type ml.g6e.2xlarge --yes
  uv run update_endpoint.py my-endpoint --region us-east-2 --instance-count 2 --yes
  uv run update_endpoint.py my-endpoint --region us-east-2 --env DEEPGRAM_ENGINE_01=max_active_requests=120 --yes
  uv run update_endpoint.py my-endpoint --region us-east-2 --model-package-arn arn:… --yes   # new version
  uv run update_endpoint.py my-endpoint --region us-east-2 --rollback-to my-endpoint --yes

Exit 0 InService after update · 1 update failed / timed out · 2 AWS error · 3 needs --yes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time

from botocore.exceptions import BotoCoreError, ClientError

from _common import (DEFAULT_INFERENCE_AMI, EXIT_NEGATIVE, EXIT_OK, add_common_args, aws_error_text,
                     classify_endpoint_failure, client, confirm_or_exit, fail_error, finish,
                     log_group_state, make_session, parse_kv_list, resolve_region, say,
                     tail_endpoint_logs)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name")
    add_common_args(p)
    p.add_argument("--instance-type", default=None)
    p.add_argument("--instance-pools", default=None, help="comma-separated, priority order, max 5")
    p.add_argument("--rung-timeout-s", type=int, default=300,
                   help="VariantInstanceProvisionTimeoutInSeconds per pool rung when --instance-pools is given (default 300)")
    p.add_argument("--instance-count", type=int, default=None)
    p.add_argument("--inference-ami-version", default=None,
                   help=f"e.g. {DEFAULT_INFERENCE_AMI}; 'none' to unpin")
    p.add_argument("--model-download-timeout-s", type=int, default=None)
    p.add_argument("--startup-timeout-s", type=int, default=None)
    p.add_argument("--async-max-concurrency", type=int, default=None)
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="REPLACES the container Environment with these entries (repeatable)")
    p.add_argument("--clear-env", action="store_true", help="remove all Environment overrides")
    p.add_argument("--model-package-arn", default=None, help="switch to another model package version")
    p.add_argument("--rollback-to", default=None, metavar="ENDPOINT_CONFIG_NAME",
                   help="UpdateEndpoint straight to an existing config (no new resources)")
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--wait-max-minutes", type=int, default=30)
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()

    session = make_session(args)
    region = resolve_region(args, session)
    sm = client(session, "sagemaker", region=region, fips=args.fips)
    logs = client(session, "logs", region=region, fips=args.fips)
    result: dict = {"endpoint_name": args.name, "region": region}

    try:
        ep = sm.describe_endpoint(EndpointName=args.name)
        cfg = sm.describe_endpoint_config(EndpointConfigName=ep["EndpointConfigName"])
    except (ClientError, BotoCoreError) as e:
        fail_error("could not describe the endpoint / its config", e)
    if ep.get("EndpointStatus") not in ("InService", "Failed"):
        say(f"endpoint is {ep.get('EndpointStatus')}; UpdateEndpoint needs InService (or Failed)")
        return finish(args, result, EXIT_NEGATIVE)
    old_cfg_name = ep["EndpointConfigName"]
    result["previous_endpoint_config"] = old_cfg_name
    pv = dict((cfg.get("ProductionVariants") or [{}])[0])
    old_model_name = pv.get("ModelName")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    if args.rollback_to:
        new_cfg_name = args.rollback_to
        try:
            sm.describe_endpoint_config(EndpointConfigName=new_cfg_name)
        except (ClientError, BotoCoreError) as e:
            fail_error(f"config {new_cfg_name} not found", e)
        changes = [f"rollback to config {new_cfg_name}"]
    else:
        try:
            model = sm.describe_model(ModelName=old_model_name)
        except (ClientError, BotoCoreError) as e:
            fail_error("could not describe the current Model", e)
        cont = dict(model.get("PrimaryContainer") or (model.get("Containers") or [{}])[0])
        changes: list[str] = []
        new_env = None
        if args.clear_env:
            new_env = {}
            changes.append("clear Environment")
        elif args.env:
            new_env = parse_kv_list(args.env, "--env")
            changes.append(f"Environment = {new_env}")
        new_arn = args.model_package_arn
        if new_arn:
            if ":model-package/" not in new_arn or new_arn.split(":")[3] != region:
                fail_error("--model-package-arn must be a ModelPackage ARN in this region")
            changes.append(f"model package → {new_arn}")

        new_model_name = old_model_name
        if new_env is not None or new_arn:
            new_model_name = f"{args.name}-{stamp}"
            new_cont = {"ModelPackageName": new_arn or cont.get("ModelPackageName")}
            env = cont.get("Environment") if new_env is None else new_env
            if env:
                new_cont["Environment"] = env
            model_kwargs = dict(ModelName=new_model_name, ExecutionRoleArn=model["ExecutionRoleArn"],
                                PrimaryContainer=new_cont, EnableNetworkIsolation=True)
        else:
            model_kwargs = None

        # copy the variant, apply overrides
        variant = {k: v for k, v in pv.items() if k in (
            "VariantName", "ModelName", "InitialInstanceCount", "InstanceType", "InstancePools",
            "VariantInstanceProvisionTimeoutInSeconds", "InferenceAmiVersion",
            "ModelDataDownloadTimeoutInSeconds", "ContainerStartupHealthCheckTimeoutInSeconds")}
        variant["ModelName"] = new_model_name
        if args.instance_type and args.instance_pools:
            fail_error("pass --instance-type or --instance-pools, not both")
        if args.instance_type:
            variant.pop("InstancePools", None); variant.pop("VariantInstanceProvisionTimeoutInSeconds", None)
            variant["InstanceType"] = args.instance_type
            changes.append(f"instance type → {args.instance_type}")
        if args.instance_pools:
            types = [t.strip() for t in args.instance_pools.split(",") if t.strip()]
            if not 1 <= len(types) <= 5:
                fail_error("--instance-pools needs 1–5 types")
            variant.pop("InstanceType", None)
            variant["InstancePools"] = [{"InstanceType": t, "Priority": i + 1} for i, t in enumerate(types)]
            variant["VariantInstanceProvisionTimeoutInSeconds"] = args.rung_timeout_s
            changes.append(f"instance pools → {types}")
        if args.instance_count is not None:
            variant["InitialInstanceCount"] = args.instance_count
            changes.append(f"instance count → {args.instance_count}")
        if args.inference_ami_version is not None:
            if args.inference_ami_version.lower() in ("none", "default", ""):
                variant.pop("InferenceAmiVersion", None); changes.append("unpin InferenceAmiVersion")
            else:
                variant["InferenceAmiVersion"] = args.inference_ami_version
                changes.append(f"InferenceAmiVersion → {args.inference_ami_version}")
        if args.model_download_timeout_s is not None:
            variant["ModelDataDownloadTimeoutInSeconds"] = args.model_download_timeout_s
            changes.append(f"download timeout → {args.model_download_timeout_s}")
        if args.startup_timeout_s is not None:
            variant["ContainerStartupHealthCheckTimeoutInSeconds"] = args.startup_timeout_s
            changes.append(f"startup timeout → {args.startup_timeout_s}")
        if not changes:
            fail_error("nothing to change — pass at least one override, or --rollback-to")

        new_cfg_name = f"{args.name}-{stamp}"
        cfg_kwargs: dict = {"EndpointConfigName": new_cfg_name, "ProductionVariants": [variant]}
        if cfg.get("AsyncInferenceConfig"):
            aic = dict(cfg["AsyncInferenceConfig"])
            if args.async_max_concurrency is not None:
                aic.setdefault("ClientConfig", {})["MaxConcurrentInvocationsPerInstance"] = args.async_max_concurrency
                changes.append(f"async max concurrency → {args.async_max_concurrency}")
            cfg_kwargs["AsyncInferenceConfig"] = aic
        if cfg.get("MetricsConfig"):
            cfg_kwargs["MetricsConfig"] = cfg["MetricsConfig"]

    say("== Update plan ==")
    say(f"  endpoint:        {args.name} ({ep.get('EndpointStatus')})")
    say(f"  current config:  {old_cfg_name}")
    say(f"  new config:      {new_cfg_name}")
    for c in changes:
        say(f"  change:          {c}")
    say("  rollout:         blue/green all-at-once (needs quota for both fleets briefly; live streams finish on the old fleet)")
    confirm_or_exit(args, f"UPDATE endpoint {args.name} in {region}: " + "; ".join(changes))

    try:
        if not args.rollback_to:
            if model_kwargs:
                sm.create_model(**model_kwargs)
                say(f"-- created Model {model_kwargs['ModelName']}")
            sm.create_endpoint_config(**cfg_kwargs)
            say(f"-- created EndpointConfig {new_cfg_name}")
        sm.update_endpoint(EndpointName=args.name, EndpointConfigName=new_cfg_name)
        say(f"-- UpdateEndpoint → {new_cfg_name} (Updating)")
    except (ClientError, BotoCoreError) as e:
        fail_error("update failed", e)
    result.update(new_endpoint_config=new_cfg_name, changes=changes,
                  rollback_command=f"update_endpoint.py {args.name} --region {region} --rollback-to {old_cfg_name} --yes")

    if args.no_wait:
        result["status"] = "Updating"
        return finish(args, result, EXIT_OK)
    deadline = time.monotonic() + args.wait_max_minutes * 60
    last = None
    while True:
        d = sm.describe_endpoint(EndpointName=args.name)
        st = d.get("EndpointStatus")
        if st != last:
            say(f"  {dt.datetime.now().strftime('%H:%M:%S')}  {st}")
            last = st
        if st == "InService" and d.get("EndpointConfigName") == new_cfg_name:
            result["status"] = st
            say(f"Updated. Previous config kept for rollback: {old_cfg_name}")
            return finish(args, result, EXIT_OK)
        if st in ("Failed", "RollingBack", "OutOfService") or time.monotonic() > deadline:
            exists, note = log_group_state(logs, args.name)
            lines = tail_endpoint_logs(logs, args.name, 40) if exists else []
            result.update(status=st, failure_reason=d.get("FailureReason"), log_group=note, log_tail=lines,
                          classification=classify_endpoint_failure(st, d.get("FailureReason"), exists, lines))
            say(f"update did not complete: {st} — {d.get('FailureReason')}")
            say(f"  cause: {result['classification']['kind']}   next: {result['classification']['next_step']}")
            say(f"  the endpoint keeps serving on {d.get('EndpointConfigName')} if SageMaker rolled back; "
                f"otherwise: {result['rollback_command']}")
            return finish(args, result, EXIT_NEGATIVE)
        time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
