#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["boto3>=1.43.49"]
# ///
"""Show a Deepgram SageMaker endpoint's status, configuration, recent container logs and a diagnosis.

  uv run endpoint_status.py my-endpoint --region us-east-2
  uv run endpoint_status.py my-endpoint --region us-east-2 --wait        # block until InService/Failed
  uv run endpoint_status.py my-endpoint --region us-east-2 --log-lines 200 --json

Reports: EndpointStatus (+ FailureReason), the variant (instance type / pools,
count, AMI, timeouts), whether the endpoint is asynchronous, whether a container
log group exists (absent ⇒ the container never started ⇒ provisioning problem,
not the image), the last log lines, and a named cause + next step for anything
that is not InService.

Exit 0 InService · 1 Failed or not InService · 2 AWS error.
"""
from __future__ import annotations

import argparse
import sys
import time

from botocore.exceptions import BotoCoreError, ClientError

from _common import (EXIT_NEGATIVE, EXIT_OK, add_common_args, aws_error_text,
                     classify_endpoint_failure, client, fail_error, finish, log_group_state,
                     make_session, resolve_region, say, tail_endpoint_logs)


def describe(sm, name: str) -> dict:
    d = sm.describe_endpoint(EndpointName=name)
    out = {"endpoint_name": name, "status": d.get("EndpointStatus"), "failure_reason": d.get("FailureReason"),
           "endpoint_config_name": d.get("EndpointConfigName"), "endpoint_arn": d.get("EndpointArn"),
           "creation_time": d.get("CreationTime"), "last_modified": d.get("LastModifiedTime"),
           "production_variants": d.get("ProductionVariants"), "async": bool(d.get("AsyncInferenceConfig"))}
    try:
        c = sm.describe_endpoint_config(EndpointConfigName=d["EndpointConfigName"])
        pv = (c.get("ProductionVariants") or [{}])[0]
        out["variant"] = {k: pv.get(k) for k in ("VariantName", "ModelName", "InstanceType", "InstancePools",
                                                  "InitialInstanceCount", "InferenceAmiVersion",
                                                  "ModelDataDownloadTimeoutInSeconds",
                                                  "ContainerStartupHealthCheckTimeoutInSeconds") if k in pv}
        out["async_config"] = c.get("AsyncInferenceConfig")
        out["metrics_config"] = c.get("MetricsConfig")
        try:
            m = sm.describe_model(ModelName=pv["ModelName"])
            cont = m.get("PrimaryContainer") or (m.get("Containers") or [{}])[0]
            out["model"] = {"model_package_arn": cont.get("ModelPackageName"),
                            "environment": cont.get("Environment"),
                            "network_isolation": m.get("EnableNetworkIsolation")}
        except (ClientError, BotoCoreError, KeyError) as e:
            out["model"] = {"error": aws_error_text(e) if isinstance(e, Exception) else str(e)}
    except (ClientError, BotoCoreError) as e:
        out["variant"] = {"error": aws_error_text(e)}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name")
    add_common_args(p)
    p.add_argument("--log-lines", type=int, default=40)
    p.add_argument("--wait", action="store_true", help="poll until InService or Failed")
    p.add_argument("--wait-max-minutes", type=int, default=30)
    args = p.parse_args()

    session = make_session(args)
    region = resolve_region(args, session)
    sm = client(session, "sagemaker", region=region, fips=args.fips)
    logs = client(session, "logs", region=region, fips=args.fips)

    deadline = time.monotonic() + args.wait_max_minutes * 60
    last = None
    while True:
        try:
            info = describe(sm, args.name)
        except ClientError as e:
            if "Could not find endpoint" in e.response.get("Error", {}).get("Message", ""):
                say(f"endpoint {args.name} does not exist in {region}")
                return finish(args, {"endpoint_name": args.name, "region": region, "status": "NotFound"}, EXIT_NEGATIVE)
            fail_error("DescribeEndpoint failed", e)
        except BotoCoreError as e:
            fail_error("DescribeEndpoint failed", e)
        status = info["status"]
        if status != last:
            say(f"{args.name}: {status}")
            last = status
        if not args.wait or status in ("InService", "Failed", "OutOfService") or time.monotonic() > deadline:
            break
        time.sleep(30)

    exists, note = log_group_state(logs, args.name)
    lines = tail_endpoint_logs(logs, args.name, args.log_lines) if exists else []
    info.update(region=region, log_group=note, log_tail=lines,
                classification=classify_endpoint_failure(status, info.get("failure_reason"), exists, lines))

    v = info.get("variant") or {}
    say(f"  config:   {info.get('endpoint_config_name')}   async={info['async']}")
    say(f"  variant:  type={v.get('InstanceType') or [pl['InstanceType'] for pl in (v.get('InstancePools') or [])]} "
        f"count={v.get('InitialInstanceCount')} ami={v.get('InferenceAmiVersion')} "
        f"download_timeout={v.get('ModelDataDownloadTimeoutInSeconds')} startup_timeout={v.get('ContainerStartupHealthCheckTimeoutInSeconds')}")
    if info.get("model"):
        say(f"  model:    {info['model']}")
    say(f"  logs:     {note}")
    for line in lines[-min(len(lines), 15):]:
        say(f"    | {line[:220]}")
    if info.get("failure_reason"):
        say(f"  FailureReason: {info['failure_reason']}")
    c = info["classification"]
    say(f"  cause: {c['kind']}")
    say(f"  next:  {c['next_step']}")
    return finish(args, info, EXIT_OK if status == "InService" else EXIT_NEGATIVE)


if __name__ == "__main__":
    sys.exit(main())
