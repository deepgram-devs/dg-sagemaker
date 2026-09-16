#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["boto3>=1.43.49"]
# ///
"""Delete a Deepgram SageMaker endpoint together with its EndpointConfig and Model.

Billing for the instances stops when the ENDPOINT is deleted; the config and
model are free but clutter the account, so the default removes all three. The
Marketplace subscription is untouched (it costs nothing while no endpoint runs).

  uv run teardown_endpoint.py my-deepgram-stt --region us-east-2 --yes
  uv run teardown_endpoint.py --prefix deepgram-test- --region us-east-2 --dry-run
  uv run teardown_endpoint.py my-deepgram-stt --region us-east-2 --mode endpoint-only --yes   # keep config+model

Safety rules baked in:
  * --dry-run prints the plan and stops; without --yes the script asks (tty) or exits 3.
  * An endpoint still Creating/Updating cannot be deleted; SageMaker refuses with a
    ValidationException. In that case the script does NOT delete the config and
    model (that would leave a live, billing endpoint with its backing resources
    gone) and exits 1 so you re-run once it settles.
  * After deleting, it re-lists endpoints/configs/models to confirm nothing remains.

Exit 0 = everything gone · 1 = something could not be deleted yet · 2 = AWS error · 3 = needs --yes.
"""
from __future__ import annotations

import argparse
import sys

from botocore.exceptions import BotoCoreError, ClientError

from _common import (EXIT_NEGATIVE, EXIT_OK, add_common_args, aws_error_text, client,
                     confirm_or_exit, fail_error, finish, make_session, resolve_region, say)

_NOT_FOUND_CODES = ("ResourceNotFound", "404")
_NOT_FOUND_MSGS = ("Could not find", "does not exist")


def is_not_found(e: ClientError) -> bool:
    """A bare ValidationException is NOT proof of absence — SageMaker uses the same
    code for 'Could not find endpoint' and 'Cannot update in-progress endpoint'."""
    err = e.response.get("Error", {})
    return err.get("Code", "") in _NOT_FOUND_CODES or any(m in err.get("Message", "") for m in _NOT_FOUND_MSGS)


def resolve_targets(sm, name: str | None, prefix: str | None) -> list[str]:
    if name:
        return [name]
    names = []
    for page in sm.get_paginator("list_endpoints").paginate(NameContains=prefix):
        for ep in page.get("Endpoints", []):
            if ep["EndpointName"].startswith(prefix):
                names.append(ep["EndpointName"])
    return sorted(set(names))


def plan_for(sm, ep_name: str) -> dict | None:
    try:
        d = sm.describe_endpoint(EndpointName=ep_name)
    except ClientError as e:
        if is_not_found(e):
            return None
        raise
    cfg = d.get("EndpointConfigName")
    models: set[str] = set()
    if cfg:
        try:
            c = sm.describe_endpoint_config(EndpointConfigName=cfg)
            models |= {pv["ModelName"] for pv in c.get("ProductionVariants", []) if pv.get("ModelName")}
        except ClientError as e:
            if not is_not_found(e):
                raise
            # Config already gone: fall back to the convention Model == Endpoint name.
            try:
                sm.describe_model(ModelName=ep_name)
                models.add(ep_name)
            except ClientError as e2:
                if not is_not_found(e2):
                    raise
    return {"endpoint": ep_name, "status": d.get("EndpointStatus", "?"), "config": cfg,
            "models": sorted(models)}


def delete_one(label: str, fn, **kw) -> bool:
    try:
        fn(**kw)
        say(f"    deleted {label}")
        return True
    except ClientError as e:
        if is_not_found(e):
            say(f"    (already gone) {label}")
            return True
        say(f"    REFUSED {label}: {aws_error_text(e)}")
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("name", nargs="?", help="exact endpoint name")
    g.add_argument("--prefix", help="every endpoint whose name starts with this")
    p.add_argument("--mode", choices=["full", "endpoint-only"], default="full",
                   help="full = endpoint + config + model(s) (default); endpoint-only keeps config+model")
    add_common_args(p)
    p.add_argument("--yes", action="store_true", help="delete without prompting")
    p.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    args = p.parse_args()

    session = make_session(args)
    region = resolve_region(args, session)
    sm = client(session, "sagemaker", region=region, fips=args.fips)

    try:
        targets = resolve_targets(sm, args.name, args.prefix)
        plans = [pl for t in targets if (pl := plan_for(sm, t))]
    except (ClientError, BotoCoreError) as e:
        fail_error("could not enumerate endpoints (this is not proof there are none)", e)

    result: dict = {"region": region, "mode": args.mode, "plan": plans, "deleted": [], "refused": []}
    if not plans:
        say(f"No matching endpoints in {region}.")
        return finish(args, result, EXIT_OK)

    say(f"== Teardown plan ({args.mode}) — {region} ==")
    for pl in plans:
        say(f"  endpoint {pl['endpoint']}  [{pl['status']}]")
        if args.mode == "full":
            say(f"    config {pl['config']}")
            for m in pl["models"]:
                say(f"    model  {m}")
    if args.dry_run:
        say("--dry-run: nothing deleted.")
        return finish(args, result, EXIT_OK)

    confirm_or_exit(args, f"DELETE {len(plans)} endpoint(s) in {region}"
                          + (" with their configs and models" if args.mode == "full" else "")
                          + ". Instance billing stops when the endpoint is deleted.")

    for pl in plans:
        say(f"-- {pl['endpoint']}")
        gone = delete_one(f"endpoint {pl['endpoint']}", sm.delete_endpoint, EndpointName=pl["endpoint"])
        if not gone:
            result["refused"].append(pl["endpoint"])
            say("    keeping its config/model — the endpoint is still live (Creating/Updating). "
                "Re-run once it reaches InService or Failed.")
            continue
        result["deleted"].append(pl["endpoint"])
        if args.mode == "full":
            if pl["config"]:
                delete_one(f"endpoint-config {pl['config']}", sm.delete_endpoint_config,
                           EndpointConfigName=pl["config"])
            for m in pl["models"]:
                delete_one(f"model {m}", sm.delete_model, ModelName=m)

    # independent verification -----------------------------------------------
    leftovers = {"endpoints": [], "configs": [], "models": []}
    try:
        for pl in plans:
            n = pl["endpoint"]
            leftovers["endpoints"] += [e["EndpointName"] for e in sm.list_endpoints(NameContains=n)["Endpoints"]
                                       if e["EndpointName"] == n and n not in result["refused"]
                                       and e.get("EndpointStatus") != "Deleting"]
            if args.mode == "full" and pl["config"]:
                leftovers["configs"] += [c["EndpointConfigName"] for c in
                                         sm.list_endpoint_configs(NameContains=pl["config"])["EndpointConfigs"]
                                         if c["EndpointConfigName"] == pl["config"] and n not in result["refused"]]
                for m in pl["models"]:
                    leftovers["models"] += [x["ModelName"] for x in sm.list_models(NameContains=m)["Models"]
                                            if x["ModelName"] == m and n not in result["refused"]]
        result["leftovers"] = leftovers
        if any(leftovers.values()):
            say(f"WARNING: resources still listed after delete (may be eventual consistency): {leftovers}")
        else:
            say("verified: nothing remains for the deleted endpoint(s)")
    except (ClientError, BotoCoreError) as e:
        result["leftovers"] = f"verification failed: {aws_error_text(e)}"
        say(f"note: post-delete verification failed ({aws_error_text(e)}) — not proof of leftovers, re-check manually")

    if result["refused"]:
        say(f"{len(result['refused'])} endpoint(s) NOT deleted and still billing: {', '.join(result['refused'])}")
        return finish(args, result, EXIT_NEGATIVE)
    return finish(args, result, EXIT_OK)


if __name__ == "__main__":
    sys.exit(main())
