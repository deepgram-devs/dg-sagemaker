#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["boto3>=1.43.49"]
# ///
"""Subscribe this AWS account to a Deepgram SageMaker product on AWS Marketplace.

Implements the documented Marketplace API flow
(https://developers.deepgram.com/docs/subscribe-aws-marketplace):

  1. SearchAgreements     — already ACTIVE?  then stop: you are subscribed.
  2. ListPurchaseOptions  — pick the offer, in this order:
                              --offer-id                      an offer you name (e.g. a private offer)
                              FIELD_DEMONSTRATION_PROGRAM     an AWS Marketplace Field Demonstration
                                                              Program (FDP) offer, if the account sees
                                                              one (skip with --no-fdp)
                              standard public offer           no PRIVATE_PRICING / FDP badge
  3. GetOffer / GetOfferTerms — proposal id + the terms (Legal, Support,
                            UsageBasedPricing, FreeTrialPricing) that will be accepted.
  4. CreateAgreementRequest — a QUOTE. Costs nothing, subscribes nothing.
  5. AcceptAgreementRequest — THE SUBSCRIBE ACTION. Only with --accept.
  6. GetAgreementEntitlements — poll until the entitlement leaves PENDING.

Subscribing creates a billing agreement but no charges start until an endpoint
is deployed; the standard public offer is usage-priced with a 14-day free trial
(once per product per account). Without --accept the script prints the terms
and the quote and exits 3 (needs confirmation) so the person can say yes.

An FDP offer is only returned for AWS accounts that AWS has enrolled in the
Field Demonstration Program (AWS field staff demonstrating Marketplace products
to customers). Under it the Deepgram software charge is not collected — the
SageMaker instance-hours still are — so it is the right offer for such an
account and the script prefers it. Its rate card shows the list price; that is
how FDP offers are published. See references/field-demonstration-program.md.

  uv run subscribe.py nova-3-mono-streaming            # show offer + terms, create quote, stop
  uv run subscribe.py nova-3-mono-streaming --accept   # subscribe
  uv run subscribe.py nova-3-mono-streaming --no-fdp --accept   # public offer even if FDP is visible
  uv run subscribe.py prod-5qjsvi7cpasyy --offer-id offer-xxxx --accept   # private offer

Exit 0 subscribed (or already was) · 1 no usable offer · 2 AWS error · 3 needs --accept.
"""
from __future__ import annotations

import argparse
import sys
import time

from botocore.exceptions import BotoCoreError, ClientError

from _common import (EXIT_NEGATIVE, EXIT_OK, FDP_DOC_URL, MARKETPLACE_REGION, add_common_args,
                     aws_error_text, client, confirm_or_exit, fail_error, find_product, finish,
                     list_purchase_options, load_catalog, make_session, product_choices, say)

OFFER_KIND_LABEL = {
    "field_demonstration": "Field Demonstration Program offer (no Deepgram software charge; "
                           "for AWS field staff demonstrating the product)",
    "private": "private offer",
    "public": "standard public offer",
}

ENTITLEMENT_PENDING = ("PENDING", "PROVISIONING_IN_PROGRESS")


def active_agreement(agree, product_id: str) -> dict | None:
    r = agree.search_agreements(catalog="AWSMarketplace", maxResults=50, filters=[
        {"name": "PartyType", "values": ["Acceptor"]},
        {"name": "AgreementType", "values": ["PurchaseAgreement"]},
        {"name": "ResourceIdentifier", "values": [product_id]},
    ])
    for a in r.get("agreementViewSummaries", []):
        if a.get("status") == "ACTIVE":
            return a
    return None


def summarize_term(wrapped: dict) -> dict:
    """Flatten one offer term into the few fields a person needs to see.

    GetOfferTerms returns `offerTerms: [{"legalTerm": {...}}, {"supportTerm": {...}}, ...]`
    — each term wrapped in a one-key object naming its kind; the inner object
    carries `id` and `type`.
    """
    t = next(iter(wrapped.values())) if len(wrapped) == 1 and isinstance(next(iter(wrapped.values())), dict) else wrapped
    ttype = t.get("type") or next(iter(wrapped.keys()), "?")
    out = {"id": t.get("id"), "type": ttype}
    if ttype == "LegalTerm":
        docs = t.get("documents") or []
        out["documents"] = [d.get("url") for d in docs if d.get("url")]
    elif ttype == "SupportTerm":
        out["refund_policy"] = (t.get("refundPolicy") or "")[:300]
    elif ttype == "FreeTrialPricingTerm":
        out["duration"] = t.get("duration")
    elif ttype == "UsageBasedPricingTerm":
        out["currency"] = t.get("currencyCode")
        out["rate_cards"] = []
        for rc in t.get("rateCards") or []:
            for row in rc.get("rateCard") or []:
                out["rate_cards"].append({"dimension": row.get("dimensionKey"), "price": row.get("price")})
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("product", help="catalog slug (e.g. nova-3-mono-streaming) or prod-… id")
    add_common_args(p, region=False)
    p.add_argument("--offer-id", default=None,
                   help="accept this specific offer (e.g. a private offer) instead of the one the "
                        "script would pick")
    p.add_argument("--no-fdp", action="store_true",
                   help="do not prefer a Field Demonstration Program offer; take the standard public "
                        "offer even if the account sees an FDP one")
    p.add_argument("--no-free-trial", action="store_true",
                   help="omit the FreeTrialPricingTerm (required if this account already used the trial)")
    p.add_argument("--accept", action="store_true",
                   help="actually accept the agreement (SUBSCRIBE). Without it: quote only, exit 3.")
    p.add_argument("--wait-minutes", type=int, default=10,
                   help="how long to poll for the entitlement to provision after accepting")
    args = p.parse_args()
    args.yes = args.accept  # confirm_or_exit reads .yes

    catalog = load_catalog()
    prod = find_product(catalog, args.product)
    if prod is None and args.product.startswith("prod-"):
        prod = {"slug": args.product, "product_id": args.product, "listing_name": "(not in kit catalog)"}
    if prod is None:
        fail_error(f"unknown product {args.product!r}. Known: {product_choices(catalog)}")
    pid = prod["product_id"]

    session = make_session(args, region=MARKETPLACE_REGION)
    agree = client(session, "marketplace-agreement", region=MARKETPLACE_REGION, fips=args.fips)
    disc = client(session, "marketplace-discovery", region=MARKETPLACE_REGION, fips=args.fips)
    result: dict = {"product": prod["slug"], "product_id": pid, "listing_name": prod["listing_name"]}

    # 1. already subscribed? -----------------------------------------------
    try:
        existing = active_agreement(agree, pid)
    except (ClientError, BotoCoreError) as e:
        fail_error("could not check existing agreements (SearchAgreements)", e)
    # 2. offers ------------------------------------------------------------
    try:
        options = list_purchase_options(disc, pid)
    except (ClientError, BotoCoreError) as e:
        if existing:  # already subscribed: the offer list is informational only
            say(f"note: ListPurchaseOptions failed ({aws_error_text(e)}); cannot tell which kind of "
                "offer the active agreement is on")
            options = []
        else:
            fail_error("ListPurchaseOptions failed", e)
    fdp_offers = [o for o in options if o["kind"] == "field_demonstration"]
    result["purchase_options"] = options
    result["fdp_offer_available"] = bool(fdp_offers)

    if existing:
        active_offer = (existing.get("proposalSummary") or {}).get("offerId")
        active_kind = next((o["kind"] for o in options if o["offer_id"] == active_offer), None)
        result.update(status="already_subscribed", agreement_id=existing["agreementId"],
                      offer_id=active_offer, offer_kind=active_kind)
        say(f"{prod['slug']} ({pid}) is ALREADY SUBSCRIBED — agreement {existing['agreementId']} "
            f"on offer {active_offer}" + (f" ({OFFER_KIND_LABEL[active_kind]})" if active_kind else "") + ".")
        if fdp_offers and active_kind not in (None, "field_demonstration"):
            say(f"NOTE: this account also sees a Field Demonstration Program offer "
                f"({fdp_offers[0]['offer_id']}) for this product, but the active subscription is on "
                f"the {OFFER_KIND_LABEL[active_kind]}. Under FDP the Deepgram software charge is not "
                f"collected. To move to it, cancel this subscription in the console "
                f"({catalog.get('manage_subscriptions_url', '')}) and re-run subscribe.py.")
            result["fdp_offer_id"] = fdp_offers[0]["offer_id"]
        say("Only one offer per listing can be active at a time. To switch offers, cancel "
            f"this subscription first: {catalog.get('manage_subscriptions_url', '')}")
        say("Next: resolve_model_package_arn.py")
        return finish(args, result, EXIT_OK)

    say(f"Purchase options for {pid}:")
    for o in options:
        say(f"  {o['offer_id']}  {o['kind']:20} badges={o['badges']}  name={o['name']!r}")
    if args.offer_id:
        chosen = next((o for o in options if o["offer_id"] == args.offer_id), None)
        if chosen is None:
            fail_error(f"--offer-id {args.offer_id} is not among this product's purchase options")
    elif fdp_offers and not args.no_fdp:
        chosen = fdp_offers[0]
        say("")
        say("This account sees a Field Demonstration Program (FDP) offer, so it is an AWS account "
            "enrolled in the program. Selecting it: the Deepgram software charge is not collected "
            "under FDP (the rate card below shows the list price; that is how FDP offers are "
            "published). SageMaker instance-hours are still billed by AWS. FDP is for AWS field "
            "staff demonstrating the product — not for production workloads. Pass --no-fdp for the "
            f"standard public offer. {FDP_DOC_URL}")
        if len(fdp_offers) > 1:
            say("note: more than one FDP offer; using the first. Pass --offer-id to pick.")
    else:
        public = [o for o in options if o["kind"] == "public"]
        if not public:
            result["status"] = "no_public_offer"
            say("No public offer found. If you were sent a private offer, pass --offer-id; "
                f"otherwise subscribe via the console: {catalog['marketplace_search_url']}")
            return finish(args, result, EXIT_NEGATIVE)
        if len(public) > 1:
            say("note: more than one public offer; using the first. Pass --offer-id to pick.")
        chosen = public[0]
        if fdp_offers:
            say("note: an FDP offer exists but --no-fdp was given; taking the standard public offer.")
    offer_id = chosen["offer_id"]
    result.update(offer_id=offer_id, offer_kind=chosen["kind"])
    say(f"Selected offer {offer_id}: {OFFER_KIND_LABEL[chosen['kind']]}")

    # 3. proposal + terms ----------------------------------------------------
    try:
        offer = disc.get_offer(offerId=offer_id)
        terms_resp = disc.get_offer_terms(offerId=offer_id)
    except (ClientError, BotoCoreError) as e:
        fail_error("GetOffer / GetOfferTerms failed", e)
    proposal_id = offer.get("agreementProposalId")
    pricing_type = (offer.get("pricingModel") or {}).get("pricingModelType")
    terms = [summarize_term(t) for t in (terms_resp.get("offerTerms") or terms_resp.get("terms") or [])]
    result.update(agreement_proposal_id=proposal_id, pricing_model=pricing_type, terms=terms)
    say(f"Offer {offer_id}: pricing model {pricing_type}, proposal {proposal_id}")
    for t in terms:
        extra = {k: v for k, v in t.items() if k not in ("id", "type")}
        say(f"  {t['type']:26} {t['id']}  {extra if extra else ''}")

    requested = [{"id": t["id"]} for t in terms
                 if t["id"] and not (args.no_free_trial and t["type"] == "FreeTrialPricingTerm")]
    if not requested:
        fail_error("offer returned no terms to accept")

    # 4. quote ---------------------------------------------------------------
    def create_request(term_ids):
        return agree.create_agreement_request(agreementProposalIdentifier=proposal_id,
                                              intent="NEW", requestedTerms=term_ids)
    try:
        req = create_request(requested)
    except ClientError as e:
        msg = e.response.get("Error", {}).get("Message", "")
        if "active agreement exists" in msg:
            fail_error("an active agreement already exists for this product (the earlier check "
                       "missed it, or it was just created). Check the console 'Manage subscriptions'.", e)
        trial_ids = [t["id"] for t in terms if t["type"] == "FreeTrialPricingTerm"]
        if trial_ids and not args.no_free_trial and e.response.get("Error", {}).get("Code") == "ValidationException":
            say(f"CreateAgreementRequest rejected with the free-trial term ({msg}). "
                "Retrying WITHOUT the free trial (already used on this account?).")
            requested = [t for t in requested if t["id"] not in trial_ids]
            result["free_trial_dropped"] = True
            try:
                req = create_request(requested)
            except (ClientError, BotoCoreError) as e2:
                fail_error("CreateAgreementRequest failed", e2)
        else:
            fail_error("CreateAgreementRequest failed", e)
    except BotoCoreError as e:
        fail_error("CreateAgreementRequest failed", e)

    request_id = req.get("agreementRequestId")
    charge = req.get("chargeSummary") or {}
    result.update(agreement_request_id=request_id, charge_summary=charge,
                  requested_term_ids=[t["id"] for t in requested])
    say("")
    say(f"Quote created: agreementRequestId={request_id}")
    say(f"  chargeSummary={charge}  (usage-priced offers quote 0.00 up front; usage is billed per request)")

    # 5. accept ----------------------------------------------------------------
    confirm_or_exit(args, f"ACCEPT the AWS Marketplace agreement for {prod['listing_name']} "
                          f"({pid}) on offer {offer_id} ({OFFER_KIND_LABEL[chosen['kind']]}). "
                          f"This subscribes the account "
                          f"{'' if not args.no_free_trial else '(no free trial) '}— no charges until an endpoint runs.",
                    flag="--accept")
    try:
        acc = agree.accept_agreement_request(agreementRequestId=request_id)
    except (ClientError, BotoCoreError) as e:
        fail_error("AcceptAgreementRequest failed", e)
    agreement_id = acc.get("agreementId")
    result.update(agreement_id=agreement_id, status="accepted")
    say(f"Accepted. agreementId={agreement_id}")

    # 6. entitlement ------------------------------------------------------------
    deadline = time.monotonic() + args.wait_minutes * 60
    ent_status = None
    while True:
        try:
            ents = agree.get_agreement_entitlements(agreementId=agreement_id).get("agreementEntitlements", [])
        except (ClientError, BotoCoreError) as e:
            say(f"  (GetAgreementEntitlements: {aws_error_text(e)}; retrying)")
            ents = []
        statuses = [e.get("status") for e in ents]
        ent_status = statuses
        if ents and not any(s in ENTITLEMENT_PENDING for s in statuses):
            break
        if time.monotonic() > deadline:
            say(f"Entitlement still {statuses} after {args.wait_minutes} min; it usually completes "
                "within a few minutes. Re-check with list_products.py before deploying.")
            break
        say(f"  entitlement {statuses or 'not yet visible'} … waiting 15s")
        time.sleep(15)
    result["entitlement_status"] = ent_status
    result["status"] = "subscribed"
    say("Subscribed. Next: resolve_model_package_arn.py " + prod["slug"] + " --region <region>")
    return finish(args, result, EXIT_OK)


if __name__ == "__main__":
    sys.exit(main())
