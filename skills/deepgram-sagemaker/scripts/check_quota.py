#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["boto3>=1.43.49"]
# ///
"""Check (and optionally request) the SageMaker endpoint quota for GPU instance types in a region.

SageMaker enforces a per-region, per-instance-type quota named
"<type> for endpoint usage". A quota of 0 — the default for GPU types in many
accounts — makes CreateEndpoint fail with ResourceLimitExceeded, and, unlike
capacity, quota does not fall back across an instance-pool ladder: one
zero-quota type fails the whole call. Quotas are approved per REGION; a raise
in the wrong region is the most common false fix.

For each type: the quota value, how many are in use right now (from the quota's
own CloudWatch usage metric, 5-minute resolution, may lag), whether there is
room for one more endpoint instance, and any increase request already in flight.

  uv run check_quota.py --region us-east-2 --product nova-3-mono-streaming
  uv run check_quota.py --region us-east-2 --instance-types ml.g6.2xlarge,ml.g6e.2xlarge
  uv run check_quota.py --region us-east-2 --instance-types ml.g6.2xlarge --request 2 --yes

Exit 0 = every type has room · 1 = at least one has no room (quota 0 or full) ·
2 = the quota lookup failed (says nothing about the quota).
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

from botocore.exceptions import BotoCoreError, ClientError

from _common import (EXIT_NEGATIVE, EXIT_OK, add_common_args, aws_error_text, client,
                     confirm_or_exit, fail_error, find_product, finish, load_catalog,
                     make_session, product_choices, resolve_region, say)

QUOTA_SUFFIX = " for endpoint usage"


def endpoint_quotas(sq) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for page in sq.get_paginator("list_service_quotas").paginate(ServiceCode="sagemaker"):
        for q in page.get("Quotas", []):
            name = q.get("QuotaName", "")
            if name.endswith(QUOTA_SUFFIX):
                out[name[: -len(QUOTA_SUFFIX)]] = q
    return out


def usage_now(cw, quota: dict) -> float | None:
    um = quota.get("UsageMetric")
    if not um:
        return None
    stat = um.get("MetricStatisticRecommendation") or "Maximum"
    end = dt.datetime.now(dt.timezone.utc)
    try:
        r = cw.get_metric_statistics(
            Namespace=um["MetricNamespace"], MetricName=um["MetricName"],
            Dimensions=[{"Name": k, "Value": v} for k, v in (um.get("MetricDimensions") or {}).items()],
            StartTime=end - dt.timedelta(minutes=15), EndTime=end, Period=300, Statistics=[stat])
    except (ClientError, BotoCoreError):
        return None
    pts = r.get("Datapoints", [])
    if not pts:
        return 0.0
    latest = max(pts, key=lambda d: d["Timestamp"])
    return float(latest.get(stat, 0.0))


def open_requests(sq) -> dict[str, dict]:
    out: dict[str, dict] = {}
    try:
        for page in sq.get_paginator("list_requested_service_quota_change_history").paginate(
                ServiceCode="sagemaker"):
            for r in page.get("RequestedQuotas", []):
                if r.get("Status") in ("PENDING", "CASE_OPENED"):
                    out[r["QuotaCode"]] = {"status": r["Status"], "desired": r.get("DesiredValue"),
                                           "case_id": r.get("CaseId")}
    except (ClientError, BotoCoreError) as e:
        say(f"note: could not list in-flight quota requests ({aws_error_text(e)})")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--instance-types", default=None, help="comma-separated ml.* types")
    p.add_argument("--product", default=None,
                   help="catalog slug: check every instance type the kit lists for that product")
    p.add_argument("--request", type=int, default=None, metavar="N",
                   help="request an increase to N for every listed type whose quota is below N")
    p.add_argument("--yes", action="store_true", help="submit --request without prompting")
    args = p.parse_args()

    catalog = load_catalog()
    types: list[str] = []
    if args.product:
        prod = find_product(catalog, args.product)
        if prod is None:
            fail_error(f"unknown product {args.product!r}. Known: {product_choices(catalog)}")
        types += catalog["instance_profiles"][prod["instance_profile"]]["supported"]
    if args.instance_types:
        types += [t.strip() for t in args.instance_types.split(",") if t.strip()]
    types = list(dict.fromkeys(types))
    if not types:
        fail_error("pass --instance-types and/or --product")

    session = make_session(args)
    region = resolve_region(args, session)
    sq = client(session, "service-quotas", region=region, fips=args.fips)
    cw = client(session, "cloudwatch", region=region, fips=args.fips)

    try:
        quotas = endpoint_quotas(sq)
    except (ClientError, BotoCoreError) as e:
        fail_error(f"could not list SageMaker quotas in {region}", e)
    pending = open_requests(sq)

    rows = []
    blocked = 0
    for t in types:
        q = quotas.get(t)
        if q is None:
            rows.append({"instance_type": t, "quota": None, "note": "no such endpoint quota in this region"})
            blocked += 1
            continue
        value = float(q.get("Value", 0))
        used = usage_now(cw, q)
        room = (value >= 1) and (used is None or used + 1 <= value)
        if not room:
            blocked += 1
        rows.append({"instance_type": t, "quota_code": q["QuotaCode"], "quota": value,
                     "in_use": used, "room_for_one_more": room,
                     "adjustable": q.get("Adjustable"),
                     "pending_request": pending.get(q["QuotaCode"]),
                     "console_url": f"https://{region}.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas/{q['QuotaCode']}"})

    human = [f"{'instance type':18} {'quota':>6} {'in use':>7}  room  pending request"]
    for r in rows:
        if r["quota"] is None:
            human.append(f"{r['instance_type']:18} {'-':>6} {'-':>7}  no    ({r['note']})")
            continue
        pend = r["pending_request"]
        pend_s = f"{pend['status']} → {pend['desired']}" if pend else "-"
        used_s = "?" if r["in_use"] is None else f"{r['in_use']:g}"
        human.append(f"{r['instance_type']:18} {r['quota']:>6g} {used_s:>7}  {'yes' if r['room_for_one_more'] else 'NO ':4} {pend_s}")

    result = {"region": region, "types": rows, "all_have_room": blocked == 0}

    if args.request is not None:
        targets = [r for r in rows if r.get("quota") is not None and r["quota"] < args.request
                   and not r["pending_request"]]
        if not targets:
            human.append("")
            human.append(f"--request {args.request}: nothing to request (all at or above {args.request}, or already pending).")
        else:
            confirm_or_exit(args, f"request a SageMaker quota increase to {args.request} for "
                                  f"{', '.join(r['instance_type'] for r in targets)} in {region} "
                                  "(opens an AWS support case; approval can take hours to days)")
            submitted = []
            for r in targets:
                try:
                    resp = sq.request_service_quota_increase(ServiceCode="sagemaker",
                                                             QuotaCode=r["quota_code"],
                                                             DesiredValue=float(args.request))
                    rq = resp.get("RequestedQuota", {})
                    submitted.append({"instance_type": r["instance_type"], "request_id": rq.get("Id"),
                                      "status": rq.get("Status"), "case_id": rq.get("CaseId")})
                    human.append(f"requested {r['instance_type']} → {args.request}: {rq.get('Status')} (id {rq.get('Id')})")
                except (ClientError, BotoCoreError) as e:
                    submitted.append({"instance_type": r["instance_type"], "error": aws_error_text(e)})
                    human.append(f"request for {r['instance_type']} FAILED: {aws_error_text(e)}")
            result["requests_submitted"] = submitted

    if blocked:
        human.append("")
        human.append(f"{blocked} type(s) have no room in {region}. Deploy on a type that does, use another "
                     "region, or raise the quota (--request N, or the console_url in --json).")
        return finish(args, result, EXIT_NEGATIVE, human)
    human.append("")
    human.append("All listed types have room for one more endpoint instance.")
    return finish(args, result, EXIT_OK, human)


if __name__ == "__main__":
    sys.exit(main())
