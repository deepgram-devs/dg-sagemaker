#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["boto3>=1.43.49"]
# ///
"""List Deepgram's SageMaker products on AWS Marketplace and this account's subscription state.

Two sources, joined on product id:
  * AWS Marketplace Discovery `SearchListings`, filtered to Deepgram's seller profile
    and SAGEMAKER_MODEL fulfillment — the live list of public listings.
  * `references/products.json` — this kit's knowledge of each listing (family,
    transport, API path, required params, instance types).

Subscription state comes from Marketplace Agreement `SearchAgreements` as the
Acceptor. Only an agreement with status ACTIVE counts as subscribed; EXPIRED /
CANCELLED / REPLACED rows are history and are reported separately.

  uv run list_products.py --json
  uv run list_products.py --product nova-3-mono-streaming     # exit 0 subscribed / 1 not

Exit 1 with --product means "not subscribed". Exit 2 means the agreement lookup
FAILED, which says nothing about whether you are subscribed — fix credentials
and re-run. Read-only; nothing here subscribes.
"""
from __future__ import annotations

import argparse
import sys

from botocore.exceptions import BotoCoreError, ClientError

from _common import (DEEPGRAM_SELLER_PROFILE_ID, EXIT_ERROR, EXIT_NEGATIVE, EXIT_OK,
                     MARKETPLACE_REGION, add_common_args, aws_error_text, client, fail_error,
                     find_product, finish, load_catalog, make_session, product_choices, say)


def search_listings(disc) -> list[dict]:
    out, token = [], None
    while True:
        kw = dict(maxResults=25, filters=[
            {"filterType": "FULFILLMENT_OPTION_TYPE", "filterValues": ["SAGEMAKER_MODEL"]},
            {"filterType": "PUBLISHER", "filterValues": [DEEPGRAM_SELLER_PROFILE_ID]},
        ])
        if token:
            kw["nextToken"] = token
        r = disc.search_listings(**kw)
        for s in r.get("listingSummaries", []):
            ents = s.get("associatedEntities") or []
            pid = (ents[0].get("product") or {}).get("productId") if ents else None
            out.append({"listing_id": s.get("listingId"), "listing_name": s.get("listingName"),
                        "product_id": pid,
                        "badges": [b.get("badgeType") for b in s.get("badges", [])]})
        token = r.get("nextToken")
        if not token:
            return out


def agreements_by_product(agree) -> dict[str, list[dict]]:
    """{product_id: [ {agreement_id, status, offer_id, start, end} ... ]} for this account."""
    out: dict[str, list[dict]] = {}
    token = None
    while True:
        kw = dict(catalog="AWSMarketplace", maxResults=50, filters=[
            {"name": "PartyType", "values": ["Acceptor"]},
            {"name": "AgreementType", "values": ["PurchaseAgreement"]},
        ])
        if token:
            kw["nextToken"] = token
        r = agree.search_agreements(**kw)
        for a in r.get("agreementViewSummaries", []):
            for res in (a.get("proposalSummary") or {}).get("resources", []) or []:
                if res.get("type") == "MachineLearningProduct":
                    out.setdefault(res["id"], []).append({
                        "agreement_id": a.get("agreementId"), "status": a.get("status"),
                        "offer_id": (a.get("proposalSummary") or {}).get("offerId"),
                        "start": a.get("startTime"), "end": a.get("endTime")})
        token = r.get("nextToken")
        if not token:
            return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, region=False)
    p.add_argument("--product", default=None,
                   help="slug or prod-id to check one product (exit 1 if not subscribed)")
    p.add_argument("--skip-live-search", action="store_true",
                   help="do not call SearchListings; use only the kit catalog")
    args = p.parse_args()

    catalog = load_catalog()
    session = make_session(args, region=MARKETPLACE_REGION)
    disc = client(session, "marketplace-discovery", region=MARKETPLACE_REGION, fips=args.fips)
    agree = client(session, "marketplace-agreement", region=MARKETPLACE_REGION, fips=args.fips)

    live: dict[str, dict] = {}
    live_error = None
    if not args.skip_live_search:
        try:
            for row in search_listings(disc):
                if row["product_id"]:
                    live[row["product_id"]] = row
        except (ClientError, BotoCoreError) as e:
            live_error = aws_error_text(e)
            say(f"note: SearchListings failed ({live_error}); continuing with the kit catalog only")

    agreements: dict[str, list[dict]] = {}
    agreements_error = None
    try:
        agreements = agreements_by_product(agree)
    except (ClientError, BotoCoreError) as e:
        agreements_error = aws_error_text(e)

    rows = []
    for prod in catalog["products"]:
        pid = prod["product_id"]
        ags = agreements.get(pid, [])
        active = [a for a in ags if a["status"] == "ACTIVE"]
        if agreements_error:
            sub = "UNKNOWN"
        else:
            sub = "SUBSCRIBED" if active else "NOT_SUBSCRIBED"
        rows.append({
            "slug": prod["slug"], "product_id": pid, "listing_name": prod["listing_name"],
            "service": prod["service"], "family": prod["family"], "transport": prod["transport"],
            "invocation_modes": prod["invocation_modes"], "api_path": prod["api_path"],
            "recommended_instance_type": catalog["instance_profiles"][prod["instance_profile"]]["recommended"],
            "subscription": sub,
            "active_agreement_id": active[0]["agreement_id"] if active else None,
            "past_agreements": [a for a in ags if a["status"] != "ACTIVE"],
            "listed_publicly": (pid in live) if live else None,
            "use_when": prod["use_when"],
        })

    unknown_live = [v for k, v in live.items() if find_product(catalog, k) is None]

    result = {
        "seller_profile_id": DEEPGRAM_SELLER_PROFILE_ID,
        "marketplace_search_url": catalog["marketplace_search_url"],
        "catalog_version": catalog["catalog_version"],
        "products": rows,
        "listings_not_in_catalog": unknown_live,
        "search_listings_error": live_error,
        "agreements_error": agreements_error,
    }

    human = [f"{'slug':24} {'product id':20} {'transport':10} {'subscription':15} listing"]
    for r in rows:
        human.append(f"{r['slug']:24} {r['product_id']:20} {r['transport']:10} {r['subscription']:15} {r['listing_name']}")
    if unknown_live:
        human.append("")
        human.append("Live listings this kit does not know (newer than the catalog?):")
        for u in unknown_live:
            human.append(f"  {u['product_id']}  {u['listing_name']}")

    if agreements_error:
        human.append("")
        human.append(f"COULD NOT READ SUBSCRIPTIONS: {agreements_error}")
        human.append("Subscription column is UNKNOWN — this is not 'not subscribed'. Fix credentials/IAM and re-run.")
        return finish(args, result, EXIT_ERROR, human)

    if args.product:
        prod = find_product(catalog, args.product)
        if prod is None:
            fail_error(f"unknown product {args.product!r}. Known: {product_choices(catalog)}")
        row = next(r for r in rows if r["slug"] == prod["slug"])
        result["selected"] = row
        if row["subscription"] == "SUBSCRIBED":
            human.append("")
            human.append(f"{prod['slug']} ({prod['product_id']}): SUBSCRIBED (agreement {row['active_agreement_id']}). "
                         "Next: resolve_model_package_arn.py")
            return finish(args, result, EXIT_OK, human)
        human.append("")
        human.append(f"{prod['slug']} ({prod['product_id']}): NOT SUBSCRIBED. Next: subscribe.py {prod['slug']}")
        return finish(args, result, EXIT_NEGATIVE, human)

    return finish(args, result, EXIT_OK, human)


if __name__ == "__main__":
    sys.exit(main())
