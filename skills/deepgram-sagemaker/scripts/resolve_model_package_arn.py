#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["boto3>=1.43.49", "certifi"]
# ///
"""Resolve a Deepgram Marketplace product + version to the ModelPackage ARN for one region.

Deploying needs the ModelPackage ARN of the product VERSION in the REGION where
the endpoint will live. The AWS docs describe finding it in the console
(Marketplace → Manage subscriptions → Configure → CLI → Version → Model ARNs).
This script does the same lookup programmatically:

  1. ListFulfillmentOptions (documented Discovery API) — the versions, their
     release notes and the recommended real-time instance type.
  2. The per-region ARNs. No documented buyer API returns them (checked
     2026-09: ListFulfillmentOptions, GetProduct, GetListing, SearchListings and
     GetAgreementEntitlements all omit them). The script therefore calls the
     same Marketplace read-only operation the console uses (`GetListingView`),
     signed with your own credentials. It is read-only and has been stable, but
     it is not a documented API: if it ever fails, follow the console steps
     printed by this script and pass the ARN to deploy_endpoint.py with
     --model-package-arn.
  3. DescribeModelPackage in the target region — verifies the ARN is visible
     to this account and returns SupportedRealtimeInferenceInstanceTypes, which
     deploy_endpoint.py uses to validate your instance choice.

  uv run resolve_model_package_arn.py nova-3-mono-streaming --region us-east-2
  uv run resolve_model_package_arn.py nova-3-mono-streaming --region us-east-2 --version "de/en/es"
  uv run resolve_model_package_arn.py prod-tnv5pm6nlcm44 --region us-east-2 --list-versions --json

With several versions and no --version the versions are listed and nothing is
selected (exit 0, "selected": null) — pick one by a substring of its title.
"""
from __future__ import annotations

import argparse
import json
import ssl
import sys
import urllib.error
import urllib.request

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.exceptions import BotoCoreError, ClientError

from _common import (EXIT_OK, MARKETPLACE_REGION, add_common_args, aws_error_text, client,
                     fail_error, find_product, finish, load_catalog, make_session,
                     product_choices, resolve_region, say)

CONSOLE_STEPS = (
    "Console fallback: https://us-east-1.console.aws.amazon.com/marketplace/subscriptions → "
    "Active subscriptions → the Deepgram product → Configure → Service: 'AWS command line "
    "interface (CLI)' → pick the Version → copy the Model ARN for your region. Then pass it to "
    "deploy_endpoint.py --model-package-arn <arn>."
)


def list_versions(disc, product_id: str) -> list[dict]:
    r = disc.list_fulfillment_options(productId=product_id)
    out = []
    for fo in r.get("fulfillmentOptions", []):
        o = fo.get("sageMakerModelFulfillmentOption") or {}
        if not o:
            continue
        out.append({
            "fulfillment_option_id": o.get("fulfillmentOptionId"),
            "version": o.get("fulfillmentOptionVersion"),
            "release_notes": o.get("releaseNotes"),
            "recommended_realtime_instance_type": (o.get("recommendation") or {}).get(
                "recommendedRealtimeInferenceInstanceType"),
        })
    return out


def listing_view_arns(session, product_id: str, fips: bool) -> dict[str, dict[str, str]]:
    """{fulfillment_option_id: {region: model_package_arn}} via the console's read-only view op."""
    import certifi
    host = f"discovery{'-fips' if fips else ''}.marketplace.{MARKETPLACE_REGION}.amazonaws.com"
    url = f"https://{host}/GetListingView"
    body = json.dumps({
        "ViewQuery": {"Name": "listingUsageByProductId", "Version": 1},
        "Parameters": json.dumps({"productId": product_id}),
        "RequestContext": {"IntegrationId": "integ-ep2h7dgzh5zbo", "Locale": "en"},
        "ExecutionSemantics": "ALL_OR_NOTHING",
    })
    creds = session.get_credentials()
    if creds is None:
        fail_error("no AWS credentials available to sign the request")
    req = AWSRequest("POST", url, data=body, headers={"Content-Type": "application/json", "Host": host})
    SigV4Auth(creds.get_frozen_credentials(), "aws-marketplace", MARKETPLACE_REGION).add_auth(req)
    prepared = req.prepare()
    ctx = ssl.create_default_context(cafile=certifi.where())
    resp = urllib.request.urlopen(
        urllib.request.Request(url, data=body.encode(), headers=dict(prepared.headers), method="POST"),
        timeout=30, context=ctx)
    view = json.loads(json.loads(resp.read())["ListingView"])
    opts = view["data"]["listing"]["listingDetail"]["usage"]["fulfillmentOptions"]
    out: dict[str, dict[str, str]] = {}
    for o in opts:
        out[o["fulfillmentOptionId"]] = {m["region"]: m["modelArn"] for m in o.get("models", [])}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("product", help="catalog slug or prod-… id")
    add_common_args(p)
    p.add_argument("--version", default=None,
                   help="case-insensitive substring of the version title, e.g. 'de/en/es' or '2026-09-10'")
    p.add_argument("--list-versions", action="store_true", help="list versions and stop")
    args = p.parse_args()

    catalog = load_catalog()
    prod = find_product(catalog, args.product)
    if prod is None and args.product.startswith("prod-"):
        prod = {"slug": args.product, "product_id": args.product, "listing_name": "(not in kit catalog)",
                "instance_profile": None}
    if prod is None:
        fail_error(f"unknown product {args.product!r}. Known: {product_choices(catalog)}")
    pid = prod["product_id"]

    session = make_session(args)
    region = resolve_region(args, session)
    disc = client(session, "marketplace-discovery", region=MARKETPLACE_REGION, fips=args.fips)

    try:
        versions = list_versions(disc, pid)
    except (ClientError, BotoCoreError) as e:
        fail_error("ListFulfillmentOptions failed", e)
    if not versions:
        fail_error(f"{pid} has no SageMaker fulfillment options (is it a SageMaker product?)")

    result: dict = {"product": prod["slug"], "product_id": pid, "listing_name": prod["listing_name"],
                    "region": region, "versions": versions, "selected": None}
    say(f"{prod['listing_name']} ({pid}) — {len(versions)} version(s):")
    for v in versions:
        say(f"  [{v['fulfillment_option_id']}] {v['version']}   recommended: {v['recommended_realtime_instance_type']}")

    # ARNs (best-effort, undocumented op) -----------------------------------
    arns: dict = {}
    arn_error = None
    try:
        arns = listing_view_arns(session, pid, args.fips)
    except urllib.error.HTTPError as e:
        arn_error = f"HTTP {e.code} from GetListingView: {e.read()[:300]!r}"
    except Exception as e:  # noqa: BLE001 — surface anything; this path is best-effort by design
        arn_error = f"{type(e).__name__}: {e}"
    if arn_error:
        result["arn_lookup_error"] = arn_error
        say(f"note: could not fetch ARNs programmatically ({arn_error}).")
        say(CONSOLE_STEPS)
    for v in versions:
        v["model_package_arns_by_region"] = arns.get(v["fulfillment_option_id"], {})

    if args.list_versions:
        return finish(args, result, EXIT_OK)

    # select a version ----------------------------------------------------------
    if args.version:
        want = args.version.lower()
        sel = [v for v in versions if want in (v["version"] or "").lower()]
        if not sel:
            fail_error(f"no version title contains {args.version!r}. Available: "
                       + "; ".join(v["version"] for v in versions))
        if len(sel) > 1:
            fail_error(f"--version {args.version!r} is ambiguous: " + "; ".join(v["version"] for v in sel))
        chosen = sel[0]
    elif len(versions) == 1:
        chosen = versions[0]
    else:
        say("")
        say("Several versions exist. Choose one with --version <substring of the title> "
            "(read the language list in each title / release notes).")
        return finish(args, result, EXIT_OK)

    arn = chosen["model_package_arns_by_region"].get(region)
    selected = {**chosen, "model_package_arn": arn}
    result["selected"] = selected
    if not arn:
        avail = sorted(chosen["model_package_arns_by_region"])
        if avail:
            fail_error(f"version {chosen['version']!r} has no ModelPackage in {region}. "
                       f"Regions available: {', '.join(avail)}")
        say("")
        say(f"No ARN available for {region}. {CONSOLE_STEPS}")
        return finish(args, result, EXIT_OK)

    # verify in the target region ----------------------------------------------
    sm = client(session, "sagemaker", region=region, fips=args.fips)
    try:
        d = sm.describe_model_package(ModelPackageName=arn)
        inf = d.get("InferenceSpecification") or {}
        selected.update(
            verified=True,
            model_package_status=d.get("ModelPackageStatus"),
            supported_realtime_instance_types=inf.get("SupportedRealtimeInferenceInstanceTypes", []),
            supported_content_types=inf.get("SupportedContentTypes", []),
        )
    except (ClientError, BotoCoreError) as e:
        selected.update(verified=False, verify_error=aws_error_text(e))
        say(f"note: DescribeModelPackage in {region} failed: {aws_error_text(e)}")
        say("      The ARN is still printed; deploy_endpoint.py will surface the real error "
            "(commonly: not subscribed, or no sagemaker:DescribeModelPackage permission).")

    say("")
    say(f"version:   {chosen['version']}")
    say(f"arn:       {arn}")
    say(f"recommended instance: {chosen['recommended_realtime_instance_type']}")
    if selected.get("supported_realtime_instance_types"):
        say(f"supported instances:  {', '.join(selected['supported_realtime_instance_types'])}")
    if prod.get("instance_profile"):
        say(f"kit note: {catalog['instance_profiles'][prod['instance_profile']]['note']}")
    say("Next: check_quota.py --region " + region + " --instance-types <the type(s) you are considering>")
    return finish(args, result, EXIT_OK)


if __name__ == "__main__":
    sys.exit(main())
