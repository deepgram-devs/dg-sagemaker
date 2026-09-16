#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["boto3>=1.43.49"]
# ///
"""Preflight: can this machine and these AWS credentials run the Deepgram-on-SageMaker flow?

Checks, in order, and reports each as ok / denied / error:
  1. local tooling: Python, botocore version, `uv`, `aws` CLI (optional)
  2. AWS identity (sts:GetCallerIdentity) and the region that will be used
  3. permission smoke tests for each phase of the flow:
       marketplace-agreement:SearchAgreements     (subscription status / subscribe)
       marketplace-discovery:SearchListings       (product discovery)
       sagemaker:ListEndpoints                    (deploy / status / teardown)
       servicequotas:ListServiceQuotas            (quota check)
       logs:DescribeLogGroups                     (failure diagnosis)
       iam:GetRole (only with --execution-role-name) (execution role)

A denied permission names the managed policy that grants it (references/iam.md).
Nothing here creates, changes or costs anything.

Exit 0 = everything needed is in place; 1 = at least one permission is denied;
2 = a call failed for a reason other than permissions (expired token, network).
"""
from __future__ import annotations

import argparse
import platform
import shutil
import sys

from botocore.exceptions import BotoCoreError, ClientError

from _common import (EXIT_ERROR, EXIT_NEGATIVE, EXIT_OK, MARKETPLACE_REGION, add_common_args,
                     aws_error_text, check_botocore, client, fail_error, finish, is_auth_problem,
                     make_session, resolve_region, say)

POLICY_HINTS = {
    "marketplace-agreement": "AWSMarketplaceManageSubscriptions (read) — subscribing also needs "
                             "aws-marketplace:CreateAgreementRequest/AcceptAgreementRequest "
                             "(AWSMarketplaceFullAccess or a custom policy)",
    "marketplace-discovery": "AWSMarketplaceManageSubscriptions",
    "sagemaker": "AmazonSageMakerFullAccess (or sagemaker:Create*/Describe*/Delete* on models, "
                 "endpoint-configs, endpoints + iam:PassRole on the execution role)",
    "service-quotas": "ServiceQuotasReadOnlyAccess (servicequotas:ListServiceQuotas); "
                      "ServiceQuotasFullAccess to request increases",
    "logs": "CloudWatchLogsReadOnlyAccess",
    "iam": "iam:GetRole / iam:CreateRole / iam:AttachRolePolicy for the execution role "
           "(IAMFullAccess, or a scoped policy on role/deepgram-sagemaker-execution)",
}


def probe(name: str, fn) -> dict:
    try:
        fn()
        return {"check": name, "status": "ok"}
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("AccessDeniedException", "AccessDenied", "UnauthorizedOperation"):
            return {"check": name, "status": "denied", "detail": aws_error_text(e)}
        if code == "NoSuchEntity":
            return {"check": name, "status": "missing", "detail": aws_error_text(e),
                    "next_step": "create it with create_execution_role.py"}
        return {"check": name, "status": "error", "detail": aws_error_text(e),
                "auth_problem": is_auth_problem(e)}
    except BotoCoreError as e:
        return {"check": name, "status": "error", "detail": aws_error_text(e),
                "auth_problem": is_auth_problem(e)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--execution-role-name", default=None,
                   help="also check iam:GetRole on this SageMaker execution role name")
    args = p.parse_args()

    result: dict = {"local": {}, "identity": {}, "checks": []}

    # 1. local tooling ------------------------------------------------------
    ok_boto, boto_ver = check_botocore()
    result["local"] = {
        "python": platform.python_version(),
        "botocore": boto_ver,
        "botocore_ok": ok_boto,
        "uv": shutil.which("uv") or None,
        "aws_cli": shutil.which("aws") or None,
    }
    say(f"python {platform.python_version()}   botocore {boto_ver}"
        f"{'' if ok_boto else '  (TOO OLD — need >= 1.43.49; uv run resolves this automatically)'}")
    say(f"uv: {result['local']['uv'] or 'NOT FOUND (install: https://docs.astral.sh/uv/)'}   "
        f"aws cli: {result['local']['aws_cli'] or 'not found (optional)'}")

    # 2. identity + region --------------------------------------------------
    session = make_session(args)
    region = resolve_region(args, session)
    try:
        ident = client(session, "sts", region=region, fips=args.fips).get_caller_identity()
    except (ClientError, BotoCoreError) as e:
        fail_error("could not resolve AWS identity", e)
    result["identity"] = {"account": ident["Account"], "arn": ident["Arn"], "region": region,
                          "profile": args.profile, "fips": args.fips}
    say(f"account {ident['Account']}   identity {ident['Arn']}")
    say(f"region  {region}{'   FIPS on' if args.fips else ''}")

    # 3. permission smoke tests -------------------------------------------
    agree = client(session, "marketplace-agreement", region=MARKETPLACE_REGION, fips=args.fips)
    disc = client(session, "marketplace-discovery", region=MARKETPLACE_REGION, fips=args.fips)
    sm = client(session, "sagemaker", region=region, fips=args.fips)
    sq = client(session, "service-quotas", region=region, fips=args.fips)
    logs = client(session, "logs", region=region, fips=args.fips)

    checks = [
        ("marketplace-agreement", "marketplace-agreement:SearchAgreements",
         lambda: agree.search_agreements(catalog="AWSMarketplace", maxResults=1, filters=[
             {"name": "PartyType", "values": ["Acceptor"]},
             {"name": "AgreementType", "values": ["PurchaseAgreement"]}])),
        ("marketplace-discovery", "marketplace-discovery:SearchListings",
         lambda: disc.search_listings(maxResults=1, filters=[
             {"filterType": "FULFILLMENT_OPTION_TYPE", "filterValues": ["SAGEMAKER_MODEL"]}])),
        ("sagemaker", f"sagemaker:ListEndpoints ({region})", lambda: sm.list_endpoints(MaxResults=1)),
        ("service-quotas", f"servicequotas:ListServiceQuotas ({region})",
         lambda: sq.list_service_quotas(ServiceCode="sagemaker", MaxResults=1)),
        ("logs", f"logs:DescribeLogGroups ({region})",
         lambda: logs.describe_log_groups(logGroupNamePrefix="/aws/sagemaker/Endpoints/", limit=1)),
    ]
    if args.execution_role_name:
        iam = client(session, "iam", region=region, fips=args.fips)
        checks.append(("iam", f"iam:GetRole {args.execution_role_name}",
                       lambda: iam.get_role(RoleName=args.execution_role_name)))

    denied = errored = missing = 0
    for service, label, fn in checks:
        r = probe(label, fn)
        if r["status"] == "denied":
            denied += 1
            r["grant_with"] = POLICY_HINTS[service]
        elif r["status"] == "error":
            errored += 1
        elif r["status"] == "missing":
            missing += 1
        result["checks"].append(r)
        mark = {"ok": "ok     ", "denied": "DENIED ", "error": "ERROR  ", "missing": "MISSING"}[r["status"]]
        say(f"  {mark} {label}" + (f"  -> {r.get('detail')}" if r["status"] != "ok" else ""))
        if r["status"] == "denied":
            say(f"           grant with: {r['grant_with']}")
        if r["status"] == "missing":
            say(f"           next: {r['next_step']}")

    if errored:
        result["verdict"] = "error"
        say("")
        say("One or more calls FAILED (not denied). Fix that first — an expired SSO token is the "
            "usual cause: aws sso login --profile <profile>")
        return finish(args, result, EXIT_ERROR)
    if denied:
        result["verdict"] = "missing_permissions"
        say("")
        say(f"{denied} permission(s) missing. See references/iam.md.")
        return finish(args, result, EXIT_NEGATIVE)
    if missing:
        result["verdict"] = "ready_but_role_missing"
        say("")
        say("Permissions are fine; the execution role does not exist yet — create_execution_role.py creates it.")
        return finish(args, result, EXIT_NEGATIVE)
    result["verdict"] = "ready"
    say("")
    say("Preflight passed. Next: list_products.py to see the Deepgram listings and subscription state.")
    return finish(args, result, EXIT_OK)


if __name__ == "__main__":
    sys.exit(main())
