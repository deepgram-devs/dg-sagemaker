#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["boto3>=1.43.49"]
# ///
"""Create (or validate) the IAM execution role SageMaker assumes to run the Deepgram container.

One role is reusable across every Deepgram endpoint in the account. It needs:
  * a trust policy allowing sagemaker.amazonaws.com to assume it
  * the managed policy AmazonSageMakerFullAccess (pulls the Marketplace model
    package, writes CloudWatch logs and metrics)
  * for ASYNCHRONOUS endpoints only: read/write on the S3 bucket that holds
    inputs, outputs and failures (--async-bucket adds an inline policy)

Idempotent: re-running attaches whatever is missing and changes nothing else.

  uv run create_execution_role.py                                 # role deepgram-sagemaker-execution
  uv run create_execution_role.py --async-bucket my-sagemaker-bucket
  uv run create_execution_role.py --existing-role-arn arn:aws:iam::123456789012:role/MyRole   # validate only

Prints the role ARN for deploy_endpoint.py --execution-role-arn. Note: a role
created seconds ago can take ~10 s to become assumable; deploy_endpoint.py
retries CreateModel briefly for that reason.
"""
from __future__ import annotations

import argparse
import json
import sys

from botocore.exceptions import BotoCoreError, ClientError

from _common import (EXIT_NEGATIVE, EXIT_OK, add_common_args, aws_error_text, client, fail_error,
                     finish, make_session, say)

DEFAULT_ROLE_NAME = "deepgram-sagemaker-execution"
SAGEMAKER_FULL_ACCESS = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Principal": {"Service": "sagemaker.amazonaws.com"},
                   "Action": "sts:AssumeRole"}],
}


def s3_policy(bucket: str) -> dict:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"],
             "Resource": f"arn:aws:s3:::{bucket}/*"},
            {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": f"arn:aws:s3:::{bucket}"},
        ],
    }


def trust_allows_sagemaker(doc: dict) -> bool:
    for st in doc.get("Statement", []):
        if st.get("Effect") != "Allow":
            continue
        svc = (st.get("Principal") or {}).get("Service")
        svcs = [svc] if isinstance(svc, str) else (svc or [])
        if "sagemaker.amazonaws.com" in svcs:
            return True
    return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, region=False)
    p.add_argument("--role-name", default=DEFAULT_ROLE_NAME)
    p.add_argument("--existing-role-arn", default=None,
                   help="validate this role instead of creating one (no changes made)")
    p.add_argument("--async-bucket", default=None,
                   help="also grant S3 read/write on this bucket (asynchronous endpoints — temporarily not "
                        "supported for Marketplace-hosted Deepgram; only with a Deepgram representative)")
    args = p.parse_args()

    # IAM is global; the region only picks the API endpoint.
    session = make_session(args, region="us-east-1")
    iam = client(session, "iam", region="us-east-1", fips=args.fips)
    result: dict = {"changes": []}

    role_name = args.role_name
    if args.existing_role_arn:
        role_name = args.existing_role_arn.rsplit("/", 1)[-1]

    try:
        role = iam.get_role(RoleName=role_name)["Role"]
        exists = True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "NoSuchEntity":
            fail_error(f"iam:GetRole {role_name} failed", e)
        exists = False
        role = None
    except BotoCoreError as e:
        fail_error(f"iam:GetRole {role_name} failed", e)

    if args.existing_role_arn:
        if not exists:
            fail_error(f"role {role_name} does not exist")
        trust_ok = trust_allows_sagemaker(role["AssumeRolePolicyDocument"])
        attached = [a["PolicyArn"] for a in iam.list_attached_role_policies(RoleName=role_name)["AttachedPolicies"]]
        result.update(role_arn=role["Arn"], role_name=role_name, trust_allows_sagemaker=trust_ok,
                      attached_policies=attached,
                      has_sagemaker_full_access=SAGEMAKER_FULL_ACCESS in attached)
        say(f"role {role['Arn']}: trust sagemaker={trust_ok}, AmazonSageMakerFullAccess={SAGEMAKER_FULL_ACCESS in attached}")
        if not trust_ok:
            say("The trust policy does not allow sagemaker.amazonaws.com to assume this role — CreateModel will fail.")
            return finish(args, result, EXIT_NEGATIVE)
        if SAGEMAKER_FULL_ACCESS not in attached:
            say("AmazonSageMakerFullAccess is not attached. The role must at least be able to read the "
                "Marketplace model package and write CloudWatch logs/metrics; verify your custom policy covers that.")
        return finish(args, result, EXIT_OK)

    if not exists:
        try:
            role = iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(TRUST_POLICY),
                                   Description="Execution role for Deepgram SageMaker endpoints")["Role"]
            result["changes"].append("created role")
            say(f"created role {role['Arn']}")
        except (ClientError, BotoCoreError) as e:
            fail_error(f"iam:CreateRole {role_name} failed", e)
    else:
        say(f"role exists: {role['Arn']}")
        if not trust_allows_sagemaker(role["AssumeRolePolicyDocument"]):
            try:
                iam.update_assume_role_policy(RoleName=role_name, PolicyDocument=json.dumps(TRUST_POLICY))
                result["changes"].append("updated trust policy to allow sagemaker.amazonaws.com")
                say("updated trust policy to allow sagemaker.amazonaws.com")
            except (ClientError, BotoCoreError) as e:
                fail_error("iam:UpdateAssumeRolePolicy failed", e)

    try:
        attached = [a["PolicyArn"] for a in iam.list_attached_role_policies(RoleName=role_name)["AttachedPolicies"]]
        if SAGEMAKER_FULL_ACCESS not in attached:
            iam.attach_role_policy(RoleName=role_name, PolicyArn=SAGEMAKER_FULL_ACCESS)
            result["changes"].append("attached AmazonSageMakerFullAccess")
            say("attached AmazonSageMakerFullAccess")
    except (ClientError, BotoCoreError) as e:
        fail_error("attaching AmazonSageMakerFullAccess failed", e)

    if args.async_bucket:
        pol_name = f"deepgram-async-s3-{args.async_bucket}"[:128]
        try:
            iam.put_role_policy(RoleName=role_name, PolicyName=pol_name,
                                PolicyDocument=json.dumps(s3_policy(args.async_bucket)))
            result["changes"].append(f"put inline policy {pol_name}")
            say(f"granted s3 Get/Put/List on {args.async_bucket} (inline policy {pol_name})")
        except (ClientError, BotoCoreError) as e:
            fail_error("iam:PutRolePolicy for the async bucket failed", e)

    result.update(role_arn=role["Arn"], role_name=role_name)
    say("")
    say(f"execution role ARN: {role['Arn']}")
    if result["changes"]:
        say("(newly created/changed roles can take ~10 s to propagate)")
    return finish(args, result, EXIT_OK)


if __name__ == "__main__":
    sys.exit(main())
