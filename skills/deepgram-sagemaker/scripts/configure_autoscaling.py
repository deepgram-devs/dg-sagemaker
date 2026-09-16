#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["boto3>=1.43.49"]
# ///
"""Attach (or remove) target-tracking auto-scaling to a REAL-TIME Deepgram SageMaker endpoint.

Registers the variant as an Application Auto Scaling target and adds a
target-tracking policy on SageMakerVariantConcurrentRequestsPerModelHighResolution
(10-second resolution; counts streaming connections and in-flight requests).

Choosing --target: ramp concurrent streams on ONE instance, find the level at
which latency is still acceptable, and set the target to 70–80% of it. Real-time
endpoints cannot scale to zero (min ≥ 1). Both fleets' worth of GPU quota is
needed briefly during scale-out.

Asynchronous endpoints are refused: auto-scaling for them is currently disabled
(https://developers.deepgram.com/docs/auto-scaling-sagemaker-async).

  uv run configure_autoscaling.py my-endpoint --region us-east-2 --min 1 --max 4 --target 8 --yes
  uv run configure_autoscaling.py my-endpoint --region us-east-2 --show
  uv run configure_autoscaling.py my-endpoint --region us-east-2 --remove --yes
"""
from __future__ import annotations

import argparse
import sys

from botocore.exceptions import BotoCoreError, ClientError

from _common import (EXIT_NEGATIVE, EXIT_OK, add_common_args, aws_error_text, client,
                     confirm_or_exit, fail_error, finish, make_session, resolve_region, say)

METRIC = "SageMakerVariantConcurrentRequestsPerModelHighResolution"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name")
    add_common_args(p)
    p.add_argument("--variant", default="AllTraffic")
    p.add_argument("--min", type=int, default=1)
    p.add_argument("--max", type=int, default=None)
    p.add_argument("--target", type=float, default=None,
                   help="target concurrent requests per instance (e.g. 8)")
    p.add_argument("--scale-out-cooldown", type=int, default=60)
    p.add_argument("--scale-in-cooldown", type=int, default=300)
    p.add_argument("--policy-name", default="deepgram-concurrency-target-tracking")
    p.add_argument("--show", action="store_true", help="print current scaling config and exit")
    p.add_argument("--remove", action="store_true", help="delete the policy and deregister the target")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()

    session = make_session(args)
    region = resolve_region(args, session)
    sm = client(session, "sagemaker", region=region, fips=args.fips)
    aas = client(session, "application-autoscaling", region=region, fips=args.fips)
    resource_id = f"endpoint/{args.name}/variant/{args.variant}"
    dim = "sagemaker:variant:DesiredInstanceCount"
    result: dict = {"endpoint_name": args.name, "region": region, "resource_id": resource_id}

    try:
        d = sm.describe_endpoint(EndpointName=args.name)
    except (ClientError, BotoCoreError) as e:
        fail_error("DescribeEndpoint failed", e)
    if d.get("AsyncInferenceConfig"):
        say("This is an ASYNCHRONOUS endpoint. Auto-scaling for asynchronous Deepgram endpoints is currently "
            "disabled because of a SageMaker platform limitation — contact your Deepgram or AWS representative. "
            "https://developers.deepgram.com/docs/auto-scaling-sagemaker-async")
        result["refused"] = "async_autoscaling_disabled"
        return finish(args, result, EXIT_NEGATIVE)
    if d.get("EndpointStatus") != "InService":
        say(f"endpoint is {d.get('EndpointStatus')}; scaling can only be configured on an InService endpoint")
        return finish(args, result, EXIT_NEGATIVE)
    variants = [v["VariantName"] for v in d.get("ProductionVariants", [])]
    if args.variant not in variants:
        fail_error(f"variant {args.variant!r} not found; endpoint has {variants}")

    def current():
        t = aas.describe_scalable_targets(ServiceNamespace="sagemaker", ResourceIds=[resource_id],
                                          ScalableDimension=dim).get("ScalableTargets", [])
        pol = aas.describe_scaling_policies(ServiceNamespace="sagemaker", ResourceId=resource_id,
                                            ScalableDimension=dim).get("ScalingPolicies", [])
        return {"targets": t, "policies": pol}

    try:
        if args.show:
            result["current"] = current()
            return finish(args, result, EXIT_OK)

        if args.remove:
            cur = current()
            confirm_or_exit(args, f"REMOVE auto-scaling from {args.name} ({len(cur['policies'])} policy, "
                                  f"{len(cur['targets'])} target)")
            for pol in cur["policies"]:
                aas.delete_scaling_policy(PolicyName=pol["PolicyName"], ServiceNamespace="sagemaker",
                                          ResourceId=resource_id, ScalableDimension=dim)
                say(f"deleted policy {pol['PolicyName']}")
            if cur["targets"]:
                aas.deregister_scalable_target(ServiceNamespace="sagemaker", ResourceId=resource_id,
                                               ScalableDimension=dim)
                say("deregistered scalable target")
            result["removed"] = True
            return finish(args, result, EXIT_OK)

        if args.max is None or args.target is None:
            fail_error("--max and --target are required (or use --show / --remove)")
        if args.min < 1:
            fail_error("--min must be >= 1: real-time endpoints cannot scale to zero")
        if args.max < args.min:
            fail_error("--max must be >= --min")

        confirm_or_exit(args, f"register {resource_id} for auto-scaling min={args.min} max={args.max} and "
                              f"add target-tracking on {METRIC} target={args.target}. Scale-out adds GPU instances "
                              "(billing + quota).")
        aas.register_scalable_target(ServiceNamespace="sagemaker", ResourceId=resource_id,
                                     ScalableDimension=dim, MinCapacity=args.min, MaxCapacity=args.max)
        say(f"registered target min={args.min} max={args.max}")
        pol = aas.put_scaling_policy(
            PolicyName=args.policy_name, ServiceNamespace="sagemaker", ResourceId=resource_id,
            ScalableDimension=dim, PolicyType="TargetTrackingScaling",
            TargetTrackingScalingPolicyConfiguration={
                "TargetValue": args.target,
                "PredefinedMetricSpecification": {"PredefinedMetricType": METRIC},
                "ScaleInCooldown": args.scale_in_cooldown, "ScaleOutCooldown": args.scale_out_cooldown,
            })
        say(f"policy {args.policy_name}: {pol.get('PolicyARN')}")
        result.update(policy_arn=pol.get("PolicyARN"), alarms=[a.get("AlarmName") for a in pol.get("Alarms", [])],
                      current=current())
        say("Watch scaling activity: aws application-autoscaling describe-scaling-activities "
            f"--service-namespace sagemaker --resource-id {resource_id} --region {region}")
        return finish(args, result, EXIT_OK)
    except (ClientError, BotoCoreError) as e:
        fail_error("Application Auto Scaling call failed", e)


if __name__ == "__main__":
    sys.exit(main())
