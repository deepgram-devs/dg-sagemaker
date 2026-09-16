---
name: deepgram-sagemaker
description: Set up and run Deepgram speech models (Nova-3 and Flux speech-to-text, Aura-2 and Flux text-to-speech) on Amazon SageMaker through AWS Marketplace. Use when someone wants to subscribe to a Deepgram SageMaker listing, find a model package ARN, pick an instance type, create or update or delete a SageMaker endpoint for Deepgram, test that a Deepgram endpoint works, configure auto-scaling, or debug a Deepgram endpoint that failed to start or returns 400s. Requires AWS credentials for the customer's account and the `uv` tool.
---

# Deepgram on Amazon SageMaker

You are walking a customer through running Deepgram in their own AWS account
on SageMaker, from Marketplace subscription to a tested endpoint. Every AWS
action is done by a script in `scripts/`. Do not hand-write `aws sagemaker
create-*`, `boto3.client(...).create_*` or Marketplace calls for anything a
script covers; the scripts encode the settings the container requires.

## Rules

1. **Scripts, not ad-hoc commands.** Run them with `uv run scripts/<name>.py …`
   from this skill's directory. `uv` installs the dependencies on first run;
   no venv setup is needed.
2. **Read exit codes literally.**
   `0` success · `1` negative result (not subscribed, quota 0, test failed) ·
   `2` the call itself FAILED (credentials, IAM, network) — this says nothing
   about the thing you were checking; fix the error and re-run ·
   `3` the action needs human confirmation — confirm, then re-run with `--yes`
   (or `--accept` for subscribe).
3. **Confirm before money or destruction.** Before `subscribe.py --accept`,
   `deploy_endpoint.py --yes`, `update_endpoint.py --yes`,
   `configure_autoscaling.py --yes`, `check_quota.py --request N --yes` and
   `teardown_endpoint.py --yes`, state what will happen (instance type, hourly
   GPU cost is theirs, what gets deleted) and get an explicit yes.
4. **Capacity is an ordered pool, chosen with the person.** Recommend
   `--instance-pools default` (the product's `default_pool`: recommended type
   first, then same-or-newer generation, older generation last) and show the
   package's supported list and the quota state before deploying. A single
   `--instance-type` is for a stated reason only (Savings Plan, measured target,
   reproducing an issue) and is recorded with `--single-type-reason`. See
   `references/instance-pools.md`. There is no silent default.
5. **Prefer `--json`** and read the result; the human text goes to stderr.
6. **Region is explicit.** Pass `--region` to every script. Model package ARNs,
   quotas and endpoints are all per-region.

## Workflow

### Phase 0 — Preflight
`uv run scripts/preflight.py --region <region> [--profile <p>] [--execution-role-name deepgram-sagemaker-execution]`
Fix anything DENIED using `references/iam.md`. A MISSING role is fine (Phase 3 creates it).

### Phase 1 — Use case → product and mode
Ask only what `references/decision-guide.md` needs: speech-to-text or
text-to-speech; live audio vs pre-recorded files; one known language vs
mixed/unknown; turn-taking voice agent (Flux) vs transcription (Nova-3);
region; rough concurrency; compliance needs (FIPS). Map the answers to one row
of `references/products.json` (the `slug`) and a mode: `streaming` or `sync`
(files ≤ 25 MB per request, real-time endpoint). **Asynchronous endpoints
(`InvokeEndpointAsync`, files up to 1 GB, scale-to-zero) are temporarily not
supported for Marketplace-hosted Deepgram** — if the person needs them, tell
them to reach out to a Deepgram representative and do not deploy one; the
scripts refuse `--async-bucket`. One endpoint serves one product.

### Phase 2 — Subscribe
`uv run scripts/list_products.py --product <slug> --region <region>`
If exit 1 (not subscribed): `uv run scripts/subscribe.py <slug>` shows the offer
terms and creates a quote (no charge). Read the terms to the person, then
`… --accept`. Private offer → `--offer-id`. Already-used free trial →
`--no-free-trial`. Subscribing costs nothing until an endpoint runs.

### Phase 3 — Parameters
1. `uv run scripts/resolve_model_package_arn.py <slug> --region <region> [--version "<substring>"] --json`
   — versions differ by language set; pick with the person. Output has the ARN
   for this region, the recommended instance type and the package's supported
   types. If the ARN lookup fails it prints the console path; then pass the ARN
   by hand to the next step.
2. `uv run scripts/check_quota.py --region <region> --instance-types <candidates>`
   — quota 0 blocks deployment; `--request N --yes` opens the increase (hours to days).
3. `uv run scripts/create_execution_role.py` — one role,
   reused for every endpoint. `--existing-role-arn` validates a customer role instead.
Present a parameter table (see `references/deploy-parameters.md`) with the
proposed pool (`default_pool` for the product, filtered to the package's
supported types, with each rung's quota) and get an explicit yes on it, or an
edited order. `deploy_endpoint.py` drops zero-quota rungs itself and says so.

### Phase 4 — Deploy
`uv run scripts/deploy_endpoint.py --region <region> --product <slug> --model-package-arn <arn> --execution-role-arn <role> --instance-pools default|A,B,C [--env K=V] --yes --json`
(`--instance-type T --single-type-reason "…"` when one type is intended.)
Waits for InService (10–20 min typical). On failure it prints FailureReason,
whether a container log group exists, the log tail and a named cause with the
next step; follow `references/troubleshooting.md`. A Failed endpoint does not
bill but should be removed with `teardown_endpoint.py`.

### Phase 5 — Validate
`uv run scripts/invoke_test.py <endpoint> --region <region> --product <slug> [--mode streaming|sync] [--language xx] [--text "…"] --json`
PASS means a real transcript (or real audio) came back. The `request` field in
the JSON is the exact shape (path, query, headers, port 8443 for streaming) to
copy into the customer's application; `references/invoke.md` has code per
language. For load and accuracy gates, point to the full drivers in this
repository (`python-stt/e2e`, `python-flux/e2e`, `python-tts/e2e`,
`python-flux-tts/e2e`).

### Phase 6 — Operate (optional)
- Auto-scaling (real-time only): `uv run scripts/configure_autoscaling.py <endpoint> --region <region> --min 1 --max N --target T --yes`
- Change instance type/count, AMI, env overrides or model version in place:
  `uv run scripts/update_endpoint.py <endpoint> --region <region> … --yes`
  (blue/green all-at-once; needs quota for both fleets briefly; `--rollback-to`).
- Health/logs any time: `uv run scripts/endpoint_status.py <endpoint> --region <region>`.
- Billing and usage metrics are in CloudWatch namespace `Deepgram/SageMakerInference`
  (`ConsumedUnits`); SageMaker's own are in `AWS/SageMaker`.

### Phase 7 — Tear down
`uv run scripts/teardown_endpoint.py <endpoint> --region <region> --yes`
Instance billing stops at endpoint deletion. The Marketplace subscription stays
and costs nothing while idle.

## When something is off
- Endpoint InService but every request 400s → almost always a request mismatch,
  not a broken endpoint: multilingual needs `language=multi`; Flux multi needs
  `model=flux-general-multi` and no `language`; a Streaming listing rejects
  synchronous `InvokeEndpoint`. `invoke_test.py` names these.
- Streaming open fails with HTTP 424 "Failed to establish WebSocket connection"
  → the container rejected the request (model/language/path); the real 400 is
  in the endpoint's CloudWatch log (`endpoint_status.py`). A client that hangs
  with no error instead is almost always missing `:8443` in the URI.
- Stuck in Creating → download timeout too low for a large bundle, or capacity.
- Container never starts, log group absent → provisioning (quota/capacity/AMI),
  not the image. Try `--instance-pools` or another region.
Full table: `references/troubleshooting.md`.

## Files
- `scripts/` — one script per step; `--help` on each.
- `references/products.json` — the listings, families, API paths, required params, instance types.
- `references/decision-guide.md`, `iam.md`, `deploy-parameters.md`, `instance-pools.md`, `invoke.md`, `troubleshooting.md`.
- `assets/spacewalk.wav` — 26 s English sample used by `invoke_test.py`.
