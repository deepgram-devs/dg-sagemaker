"""Shared helpers for the Deepgram-on-SageMaker onboarding scripts.

Every script in this directory imports from here. Conventions the whole kit
follows:

  Exit codes
    0  success / positive result
    1  NEGATIVE result — the check ran and the answer is "no" (not subscribed,
       quota is 0, endpoint Failed, test did not pass)
    2  ERROR — an AWS call or the environment failed. This is NOT evidence of a
       negative result. Expired credentials, missing IAM permission, wrong
       region, network. Fix the error and re-run; do not conclude anything.
    3  NEEDS CONFIRMATION — the action costs money or is destructive and the
       script was run without --yes in a non-interactive shell. Confirm with the
       human, then re-run with --yes.

  Output
    Progress and human-readable text go to STDERR. With --json the final result
    is a single JSON document on STDOUT (so it can be captured and parsed);
    without --json a readable rendering of the same result goes to STDOUT.

  Credentials / region
    Standard AWS credential chain (AWS_PROFILE, env vars, SSO, instance role).
    --profile selects a named profile. --region is REQUIRED unless AWS_REGION /
    AWS_DEFAULT_REGION or the profile carries one — there is no built-in
    default region, because ModelPackage ARNs and quotas are per-region and a
    silent default is how endpoints end up in the wrong place.

  FIPS
    --fips applies `use_fips_endpoint` PER CLIENT. It is deliberately not done
    by exporting AWS_USE_FIPS_ENDPOINT for the whole process: that also
    FIPS-ifies the IAM Identity Center (SSO) credential resolver, whose FIPS
    hostname does not exist, and the failure then looks intermittent because a
    warm credential cache masks it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
    ProfileNotFound,
    UnauthorizedSSOTokenError,
)

EXIT_OK = 0
EXIT_NEGATIVE = 1
EXIT_ERROR = 2
EXIT_NEEDS_CONFIRMATION = 3

#: The AWS Marketplace control-plane APIs (Discovery, Agreement) live in
#: us-east-1 only, regardless of where the endpoint will be deployed.
MARKETPLACE_REGION = "us-east-1"
#: Deepgram's AWS Marketplace seller profile id (public; used to filter listings).
DEEPGRAM_SELLER_PROFILE_ID = "6efa21f9-9a33-4cae-ba44-756436fa71dd"
#: Minimum botocore that knows the MetricsConfig.EnableDetailedObservability
#: field and the marketplace-discovery service.
MIN_BOTOCORE = (1, 43, 49)

#: The inference AMI Deepgram's current model packages require (NVIDIA driver
#: 580 / CUDA 13). See references/deploy-parameters.md.
DEFAULT_INFERENCE_AMI = "al2023-ami-sagemaker-inference-gpu-4-1"
INFERENCE_AMI_DRIVERS = {
    "al2-ami-sagemaker-inference-gpu-2": 535,
    "al2-ami-sagemaker-inference-gpu-2-1": 535,
    "al2-ami-sagemaker-inference-gpu-3-1": 550,
    "al2023-ami-sagemaker-inference-gpu-4-1": 580,
}

SKILL_DIR = Path(__file__).resolve().parent.parent
PRODUCTS_JSON = SKILL_DIR / "references" / "products.json"
ASSETS_DIR = SKILL_DIR / "assets"


# ---------------------------------------------------------------------------
# argparse + session plumbing
# ---------------------------------------------------------------------------

def add_common_args(p: argparse.ArgumentParser, *, region: bool = True) -> None:
    """--profile, --fips, --json, and (when `region`) --region."""
    if region:
        p.add_argument("--region", default=None,
                       help="AWS region of the endpoint (required unless AWS_REGION / "
                            "AWS_DEFAULT_REGION or the profile sets one)")
    p.add_argument("--profile", default=None, help="AWS named profile (optional)")
    p.add_argument("--fips", action="store_true",
                   help="use the FIPS 140-3 AWS endpoints for every call this script makes "
                        "(applied per client; not every region has FIPS endpoints)")
    p.add_argument("--json", action="store_true",
                   help="print the final result as one JSON document on stdout")


def say(msg: str = "") -> None:
    """Progress / human text. Always stderr so a --json stdout stays clean."""
    print(msg, file=sys.stderr, flush=True)


def make_session(args: argparse.Namespace, *, region: str | None = None) -> boto3.Session:
    profile = getattr(args, "profile", None) or None
    try:
        return boto3.Session(profile_name=profile, region_name=region)
    except ProfileNotFound as e:
        fail_error(f"AWS profile not found: {e}")


def resolve_region(args: argparse.Namespace, session: boto3.Session | None = None) -> str:
    """Explicit --region, else the environment / profile. Never a hardcoded default."""
    r = getattr(args, "region", None) or os.environ.get("AWS_REGION") \
        or os.environ.get("AWS_DEFAULT_REGION")
    if not r and session is not None:
        r = session.region_name
    if not r:
        fail_error("no AWS region: pass --region, or set AWS_REGION / AWS_DEFAULT_REGION. "
                   "(ModelPackage ARNs, quotas and endpoints are all per-region.)")
    return r


def client(session: boto3.Session, service: str, *, region: str | None = None,
           fips: bool = False, retries: int = 6):
    cfg = BotoConfig(retries={"max_attempts": retries, "mode": "adaptive"},
                     use_fips_endpoint=True if fips else None)
    return session.client(service, region_name=region, config=cfg)


def check_botocore() -> tuple[bool, str]:
    import botocore
    ver = tuple(int(x) for x in botocore.__version__.split(".")[:3])
    ok = ver >= MIN_BOTOCORE
    return ok, botocore.__version__


# ---------------------------------------------------------------------------
# result / exit helpers
# ---------------------------------------------------------------------------

def aws_error_text(e: BaseException) -> str:
    if isinstance(e, ClientError):
        err = e.response.get("Error", {})
        op = e.operation_name
        return f"{err.get('Code', 'ClientError')} from {op}: {err.get('Message', str(e))}"
    return f"{type(e).__name__}: {e}"


def is_auth_problem(e: BaseException) -> bool:
    if isinstance(e, (NoCredentialsError, PartialCredentialsError, UnauthorizedSSOTokenError)):
        return True
    if isinstance(e, ClientError):
        code = e.response.get("Error", {}).get("Code", "")
        return code in ("ExpiredToken", "ExpiredTokenException", "InvalidClientTokenId",
                        "UnrecognizedClientException", "AccessDeniedException", "AccessDenied",
                        "UnauthorizedOperation", "InvalidIdentityToken")
    return "Token has expired" in str(e) or "Unable to locate credentials" in str(e)


def fail_error(msg: str, exc: BaseException | None = None) -> None:
    """Exit 2. Loud about the fact that this is NOT a negative finding."""
    say("")
    say("ERROR (this is not a negative result — the call itself failed):")
    say(f"  {msg}")
    if exc is not None:
        say(f"  {aws_error_text(exc)}")
        if is_auth_problem(exc):
            say("  Looks like a credentials/permission problem. Typical fixes:")
            say("    aws sso login --profile <profile>     (expired IAM Identity Center token)")
            say("    check AWS_PROFILE / AWS_ACCESS_KEY_ID  (wrong or missing credentials)")
            say("    attach the IAM policy named in references/iam.md for this step")
    sys.exit(EXIT_ERROR)


def fail_negative(msg: str) -> None:
    say("")
    say(f"RESULT: {msg}")
    sys.exit(EXIT_NEGATIVE)


def finish(args: argparse.Namespace, result: dict, rc: int = EXIT_OK,
           human: list[str] | None = None) -> int:
    """Emit the final result and return the exit code."""
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, default=str))
        if human:
            for line in human:
                say(line)
    else:
        if human:
            for line in human:
                print(line)
        else:
            print(json.dumps(result, indent=2, default=str))
    return rc


def confirm_or_exit(args: argparse.Namespace, what: str, flag: str = "--yes") -> None:
    """Gate for money-costing or destructive actions.

    --yes proceeds. Otherwise, in a terminal, ask; in a non-interactive shell
    exit 3 so the calling agent knows to confirm with the human and re-run.
    """
    if getattr(args, "yes", False):
        return
    say("")
    say(f"ABOUT TO: {what}")
    if sys.stdin.isatty():
        ans = input("Proceed? [y/N] ").strip().lower()
        if ans in ("y", "yes"):
            return
        say("aborted")
        sys.exit(EXIT_NEGATIVE)
    say(f"Refusing to proceed without {flag} in a non-interactive shell.")
    say(f"Confirm this action with the person you are working for, then re-run with {flag}.")
    sys.exit(EXIT_NEEDS_CONFIRMATION)


def parse_kv_list(items: list[str], flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for kv in items or []:
        if "=" not in kv:
            fail_error(f"{flag} expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        out[k.strip()] = v
    return out


# ---------------------------------------------------------------------------
# product catalog
# ---------------------------------------------------------------------------

def load_catalog() -> dict:
    with open(PRODUCTS_JSON, encoding="utf-8") as f:
        return json.load(f)


def find_product(catalog: dict, key: str) -> dict | None:
    """Look a product up by slug or by product id (prod-...)."""
    key = key.strip()
    for prod in catalog["products"]:
        if key == prod["slug"] or key == prod["product_id"] or key in prod.get("aliases", []):
            return prod
    return None


def product_choices(catalog: dict) -> str:
    return ", ".join(p["slug"] for p in catalog["products"])


# ---------------------------------------------------------------------------
# endpoint diagnostics shared by deploy / status
# ---------------------------------------------------------------------------

def log_group_state(logs, endpoint_name: str) -> tuple[bool | None, str]:
    """(exists, note). `None` means the lookup itself failed — unknown, not absent."""
    name = f"/aws/sagemaker/Endpoints/{endpoint_name}"
    try:
        r = logs.describe_log_groups(logGroupNamePrefix=name)
    except (ClientError, BotoCoreError) as e:
        return None, f"could not query CloudWatch Logs ({aws_error_text(e)}) — unknown, not absent"
    for g in r.get("logGroups", []):
        if g.get("logGroupName") == name:
            return True, name
    return False, f"no log group {name}"


def tail_endpoint_logs(logs, endpoint_name: str, limit: int = 60) -> list[str]:
    name = f"/aws/sagemaker/Endpoints/{endpoint_name}"
    try:
        r = logs.filter_log_events(logGroupName=name, limit=limit, interleaved=True)
    except (ClientError, BotoCoreError):
        return []
    events = sorted(r.get("events", []), key=lambda e: e.get("timestamp", 0))
    return [e.get("message", "").rstrip() for e in events[-limit:]]


def classify_endpoint_failure(status: str, failure_reason: str | None,
                              log_group_exists: bool | None,
                              log_lines: list[str]) -> dict:
    """Turn a Failed/stuck endpoint into a named cause + concrete next step.

    See references/troubleshooting.md for the reasoning behind each bucket.
    """
    fr = failure_reason or ""
    joined = "\n".join(log_lines)
    if status == "InService":
        return {"kind": "healthy", "next_step": "Endpoint is InService. Run invoke_test.py."}
    if "ResourceLimitExceeded" in fr or "quota" in fr.lower():
        return {"kind": "quota",
                "next_step": "Service quota for this instance type in this region is 0 or "
                             "exhausted. Run check_quota.py and request an increase, or pick "
                             "another type/region."}
    if "InsufficientInstanceCapacity" in fr or "capacity" in fr.lower():
        return {"kind": "capacity",
                "next_step": "AWS had no capacity for this instance type here. Retry with "
                             "--instance-pools (several types) or another region."}
    if "cuda-preflight" in joined or "NVIDIA driver" in joined and "InferenceAmiVersion" in joined:
        return {"kind": "ami_driver",
                "next_step": f"The container refused the host GPU driver. Redeploy with "
                             f"--inference-ami-version {DEFAULT_INFERENCE_AMI} (the console "
                             "cannot set this field; use update_endpoint.py or deploy_endpoint.py)."}
    if "not subscribed" in fr.lower() or "Caller is not subscribed" in fr:
        return {"kind": "not_subscribed",
                "next_step": "This account is not subscribed to the Marketplace product. "
                             "Run subscribe.py, or accept a private offer in the console."}
    if "Environment variable map cannot be specified" in fr:
        return {"kind": "env_not_allowed",
                "next_step": "This listing does not yet allow container Environment overrides. "
                             "Remove --env, or ask Deepgram support to enable it for the listing."}
    if "not supported" in fr and "instance" in fr.lower():
        return {"kind": "instance_type_not_supported",
                "next_step": "Instance type is not in the model package's supported list. "
                             "Run resolve_model_package_arn.py to see the supported types."}
    if log_group_exists is False:
        return {"kind": "container_never_started",
                "next_step": "No container log group exists, so the container never launched: "
                             "a provisioning-level failure (capacity/quota/AMI), not the image. "
                             "Retry with --instance-pools or in another region."}
    if log_group_exists is True:
        if "ModelDataDownloadTimeout" in fr or "download" in fr.lower():
            return {"kind": "model_download_timeout",
                    "next_step": "Model download exceeded ModelDataDownloadTimeoutInSeconds. "
                                 "Redeploy with a larger --model-download-timeout-s (e.g. 1800)."}
        return {"kind": "container_failed",
                "next_step": "The container started and then failed. Read the log tail above; "
                             "share it with Deepgram support along with the endpoint name and region."}
    return {"kind": "unknown",
            "next_step": "Could not classify. Run endpoint_status.py NAME for logs and FailureReason."}
