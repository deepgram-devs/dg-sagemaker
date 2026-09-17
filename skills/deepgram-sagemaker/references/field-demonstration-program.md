# AWS Marketplace Field Demonstration Program (FDP) offers

Reference: https://docs.aws.amazon.com/marketplace/latest/userguide/field-demonstration-program.html

## What it is

The Field Demonstration Program lets approved AWS field employees (solutions
architects, sales and marketing staff, and similar roles) use Marketplace
products at no software charge, to demonstrate product capabilities for
education and for potential inclusion in customer workloads. SageMaker model
packages are one of the supported product types, and Deepgram's SageMaker
listings carry FDP offers.

Two things follow for this kit:

- **Only an enrolled AWS account sees the offer.** Enrollment is done by AWS
  Marketplace on the account, not by anything in this kit. For every other
  account `ListPurchaseOptions` simply does not return the FDP offer, and the
  scripts behave exactly as before (public offer, or `--offer-id` for a private
  offer).
- **The software charge is not collected; infrastructure still is.** The
  Deepgram license dimension is not billed under FDP. SageMaker instance-hours
  for the endpoint are AWS infrastructure and are billed to the account as
  usual. Tear the endpoint down after the demonstration (`teardown_endpoint.py`).

FDP is for demonstrations by AWS field staff. It is not a path for a customer's
production workload, which uses the public offer, a private offer, or the free
trial.

## How it shows up in the Marketplace API

`ListPurchaseOptions` returns the FDP offer as one more purchase option on the
product, next to the public offer:

| Field | Value |
|---|---|
| `purchaseOptionName` / `offerName` | `Field Demonstration Program Offer` |
| `badges[].badgeType` | `FIELD_DEMONSTRATION_PROGRAM` |
| `pricingModel` | `USAGE`, same as the public offer |
| `GetOfferTerms` | Legal, Support, UsageBasedPricing and FreeTrialPricing terms, same shapes as the public offer |

The rate card on the FDP offer shows the **list price** (the same
`inference.count.m.i.c` dimension as the public offer). That is how FDP offers
are published; do not read it as "this offer will be charged". The quote from
`CreateAgreementRequest` is `0.00` up front, as for any usage-priced offer.

Note that botocore's enum for purchase-option badges may not list
`FIELD_DEMONSTRATION_PROGRAM` yet; the API returns it regardless, and the kit
matches on the string.

## What the scripts do

- `list_products.py` calls `ListPurchaseOptions` for every Deepgram product and
  prints an `offer` column: `fdp,public` (kinds the account can accept) for an
  unsubscribed product, `on fdp` / `on public` / `on private` for a subscribed
  one. When any FDP offer is visible it says so and flags products that are
  subscribed on the public offer although an FDP offer exists. JSON:
  `fdp_offers_visible`, and per product `offers_available`, `fdp_offer_id`,
  `active_offer_id`, `active_offer_kind`. `--skip-offers` leaves the lookup out.
- `subscribe.py` picks the offer in this order: `--offer-id` if given → the
  FDP offer if the account sees one and `--no-fdp` was not passed → the standard
  public offer. It says which kind it selected, repeats it in the `--accept`
  confirmation, and returns `offer_kind` and `fdp_offer_available` in JSON.
  For an already-subscribed product it reports which kind of offer the active
  agreement is on and, if an FDP offer exists but the agreement is on another
  offer, says how to move.
- Everything after subscribing (`resolve_model_package_arn.py`,
  `deploy_endpoint.py`, `invoke_test.py`, …) is identical: the FDP offer
  entitles the account to the same model packages.

## Rules for the agent

1. **Say which offer is being accepted.** When `subscribe.py` selects the FDP
   offer, tell the person: this account is enrolled in the AWS Marketplace Field
   Demonstration Program; the Deepgram software charge is not collected; the
   SageMaker instance-hours are; the offer is meant for demonstrations. Then get
   the explicit yes as usual (rule 3 in SKILL.md).
2. **Do not try to "get" FDP for an account.** If no FDP offer is visible, the
   account is not enrolled. This kit cannot change that; the person's AWS
   internal guidance covers enrollment. Proceed with the public offer (free
   trial applies once per product per account) or a private offer.
3. **One active agreement per product.** An account already subscribed on the
   public offer keeps that agreement; to move to the FDP offer the person
   cancels the subscription in the console (Manage subscriptions) and re-runs
   `subscribe.py`. Do not cancel anything for them.
4. **Not a production offer.** If the person describes a production workload,
   point them at the public offer or a private offer (`--no-fdp`,
   `--offer-id`), even on an enrolled account.
