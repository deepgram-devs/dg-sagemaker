# IAM for Deepgram on SageMaker

Two identities are involved:

1. **The operator identity** — the credentials the scripts run with (a person's
   SSO role, an IAM user, a CI role). It subscribes, creates and deletes
   SageMaker resources and invokes the endpoint.
2. **The SageMaker execution role** — assumed by SageMaker to run the container.
   Created by `create_execution_role.py`.

## Operator identity, by phase

| Phase / script | Actions | Managed policy that grants them |
|---|---|---|
| `preflight.py` | `sts:GetCallerIdentity` | (always allowed) |
| `list_products.py` | `aws-marketplace:SearchAgreements`, `aws-marketplace:ViewSubscriptions`, Discovery `SearchListings`/`GetProduct`/`ListFulfillmentOptions` | `AWSMarketplaceManageSubscriptions` for discovery; agreement reads need `aws-marketplace:SearchAgreements` (in `AWSMarketplaceFullAccess`, or add it) |
| `subscribe.py` | `aws-marketplace:ListPurchaseOptions`, `GetOffer`, `GetOfferTerms`, `CreateAgreementRequest`, `AcceptAgreementRequest`, `DescribeAgreement`, `GetAgreementEntitlements`, `Subscribe` | `AWSMarketplaceFullAccess`, or `AWSMarketplaceManageSubscriptions` **plus** the Agreement Service actions (`aws-marketplace:CreateAgreementRequest`, `AcceptAgreementRequest`, `DescribeAgreement`, `SearchAgreements`, `GetAgreementTerms`, `GetAgreementEntitlements`) |
| `resolve_model_package_arn.py` | Discovery reads above; `sagemaker:DescribeModelPackage` | `AWSMarketplaceManageSubscriptions` + `AmazonSageMakerFullAccess` (or `sagemaker:DescribeModelPackage`) |
| `check_quota.py` | `servicequotas:ListServiceQuotas`, `servicequotas:ListRequestedServiceQuotaChangeHistory`, `cloudwatch:GetMetricStatistics`; `--request`: `servicequotas:RequestServiceQuotaIncrease` | `ServiceQuotasReadOnlyAccess` (+ `ServiceQuotasFullAccess` to request) |
| `create_execution_role.py` | `iam:GetRole`, `iam:CreateRole`, `iam:AttachRolePolicy`, `iam:ListAttachedRolePolicies`, `iam:PutRolePolicy`, `iam:UpdateAssumeRolePolicy` | `IAMFullAccess`, or a policy scoped to `arn:aws:iam::<acct>:role/deepgram-sagemaker-execution` |
| `deploy_endpoint.py`, `update_endpoint.py` | `sagemaker:CreateModel`, `CreateEndpointConfig`, `CreateEndpoint`, `UpdateEndpoint`, `Describe*`; **`iam:PassRole` on the execution role**; `logs:DescribeLogGroups`, `logs:FilterLogEvents` | `AmazonSageMakerFullAccess` + `iam:PassRole` on the role + `CloudWatchLogsReadOnlyAccess` |
| `endpoint_status.py` | `sagemaker:Describe*`, `logs:DescribeLogGroups`, `logs:FilterLogEvents` | `AmazonSageMakerReadOnly` + `CloudWatchLogsReadOnlyAccess` |
| `invoke_test.py` | `sagemaker:InvokeEndpoint`, `sagemaker:InvokeEndpointAsync`, **`sagemaker:InvokeEndpointWithBidirectionalStream`**, `sagemaker:DescribeEndpoint`; async: `s3:PutObject`/`GetObject` on the bucket | `AmazonSageMakerFullAccess` (includes the invoke actions) + S3 access to the async bucket |
| `configure_autoscaling.py` | `application-autoscaling:RegisterScalableTarget`, `PutScalingPolicy`, `Describe*`, `DeleteScalingPolicy`, `DeregisterScalableTarget`; SageMaker creates the service-linked role automatically | `AmazonSageMakerFullAccess` covers the application-autoscaling actions for SageMaker |
| `teardown_endpoint.py` | `sagemaker:DeleteEndpoint`, `DeleteEndpointConfig`, `DeleteModel`, `List*`, `Describe*` | `AmazonSageMakerFullAccess` |

Least-privilege production application identity (the thing that calls the
endpoint, not the operator): only `sagemaker:InvokeEndpoint`,
`sagemaker:InvokeEndpointWithBidirectionalStream` and/or
`sagemaker:InvokeEndpointAsync` on the endpoint ARN.

## Execution role (`create_execution_role.py`)

Trust policy:
```json
{"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
  "Principal": {"Service": "sagemaker.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
```
Attached: `arn:aws:iam::aws:policy/AmazonSageMakerFullAccess` — lets SageMaker
pull the Marketplace model package and write CloudWatch logs and metrics.

(Only with a Deepgram representative — asynchronous endpoints are temporarily
not supported for Marketplace-hosted Deepgram.) Asynchronous endpoints add an inline policy for the S3 bucket that holds inputs
and outputs:
```json
{"Version": "2012-10-17", "Statement": [
  {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"], "Resource": "arn:aws:s3:::<bucket>/*"},
  {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": "arn:aws:s3:::<bucket>"}]}
```
(`AmazonSageMakerFullAccess` already grants S3 access to buckets whose name
contains `sagemaker`; the inline policy is what makes any other bucket work.)

The container runs under network isolation and receives **no** AWS credentials;
the execution role is used by the SageMaker service, not by Deepgram's code.

## Symptoms → missing permission

| Symptom | Missing |
|---|---|
| `AccessDeniedException` from `SearchAgreements` | `aws-marketplace:SearchAgreements` |
| `CreateModel`: "is not authorized to perform: iam:PassRole" | `iam:PassRole` on the execution role for the operator |
| `CreateModel`: "Could not assume role" | role trust policy lacks `sagemaker.amazonaws.com`, or the role was created seconds ago (retry) |
| `CreateModel`: "Caller is not subscribed" | not IAM — subscribe first |
| Streaming client hangs, sync returns `AccessDeniedException` | `sagemaker:InvokeEndpointWithBidirectionalStream` (streaming) / `sagemaker:InvokeEndpoint` |
| Async result never appears; failure file says access denied | execution role lacks S3 access to the bucket |
