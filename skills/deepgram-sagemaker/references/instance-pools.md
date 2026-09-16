# Instance pools: the default way to choose capacity

**Recommendation: deploy with an ordered instance pool, not a single type.**
`deploy_endpoint.py --instance-pools A,B,C` (or `--instance-pools default` with
`--product`) makes SageMaker try `A`, and fall back to `B`, then `C`, when a type
has no capacity in that Availability Zone at provisioning time. A single
`--instance-type` has no fallback: when AWS is short of that GPU the endpoint
goes `Failed` with `Request to service failed` and no container log, typically
3 minutes in. This happens routinely for popular GPU types.

## How to order the pool (priority = position in the list)

1. **Priority 1: the listing's recommended type** (`recommended` in
   `products.json`, also printed by `resolve_model_package_arn.py`). It is the
   type Deepgram validated the model on and the best price/performance.
2. **Then same-generation or newer types with similar per-instance capacity**
   (`g6` → `g6e` → `g7`/`g7e`). Keeping capacity similar matters if you
   auto-scale: the predefined scaling metrics are per instance and do not
   account for a mixed fleet, so wildly different rungs make `--target` wrong
   for part of the fleet.
3. **Older generations last, as insurance only** (`g5`, and `g4dn` where
   supported). They work, at higher latency per stream.
4. **Never include a type the product does not support**: `g4dn` for Flux,
   `g5`/`g4dn` for Flux TTS, any single-GPU type for Aura-2. `deploy_endpoint.py`
   checks the pool against the package's `SupportedRealtimeInferenceInstanceTypes`.
5. **Use all five slots.** SageMaker allows at most five pool members; the
   `default_pool` for each family fills them (or lists every supported type when
   fewer exist), because GPU capacity shortages were the dominant deploy
   failure in testing and each extra rung is another chance to land. The
   script removes rungs the package version does not list and rungs with zero
   or exhausted quota, so the pool shrinks safely per region.
6. **Per-rung timeout.** SageMaker tries each rung for
   `VariantInstanceProvisionTimeoutInSeconds` before moving to the next;
   `deploy_endpoint.py --rung-timeout-s` sets it (default 300, AWS allows
   60–3600). A three-rung pool can therefore stay `Creating` for up to three
   times that before it fails. Note that today's most common failure
   (`Request to service failed`, no log group, ~3 minutes in) is SageMaker
   refusing up front, not a rung timing out — pools help with
   `InsufficientInstanceCapacity`-style rejections, and another region is the
   fallback for the former.

Default pools per product family (`default_pool` in `products.json`):

| Family | Default pool (priority order) |
|---|---|
| Nova-3 | `ml.g6.2xlarge`, `ml.g6e.2xlarge`, `ml.g5.2xlarge`, `ml.g4dn.2xlarge`, `ml.g7.2xlarge` |
| Flux | `ml.g6.2xlarge`, `ml.g6e.2xlarge`, `ml.g5.2xlarge`, `ml.g7.2xlarge`, `ml.g7e.2xlarge` |
| Aura-2 | `ml.g6.12xlarge`, `ml.g6e.12xlarge`, `ml.g5.12xlarge`, `ml.g4dn.12xlarge`, `ml.g7.12xlarge` |
| Flux TTS | `ml.g6e.2xlarge`, `ml.g6.2xlarge`, `ml.g7.2xlarge`, `ml.g7e.2xlarge` (all four supported types) |

Nova-3 and Aura-2 support six types; the fifth slot goes to `g7` over `g7e`
because `g7e` quota is the one most often still at 0. Swap it in if your
account has `g7e` quota and no `g4dn`.

## Quota does not fall back — capacity does

`InstancePools` retries the next rung when a type is **capacity**-constrained.
It does **not** help with **quota**: SageMaker validates the quota of every
type in the pool when the endpoint is created, so one type with quota 0 fails
the whole `CreateEndpoint`. Therefore:

- `deploy_endpoint.py` reads the endpoint quotas before creating and **drops
  zero-quota rungs** from the pool, printing which ones and why. If the quota
  lookup itself fails, the pool is used as given (the check must never be the
  reason a deploy cannot start).
- Run `check_quota.py --region R --instance-types A,B,C` first to see room and
  open increases (`--request N --yes`) for rungs you want to keep.

## When a single type is the right call

- The customer has an **ML Savings Plan** or reservation on a specific type.
- They measured a concurrency target on a specific GPU and auto-scale on it.
- Reproducing a type-specific problem.

State the reason when choosing `--instance-type`; the script prints a note
recommending pools otherwise.

## Changing the pool later

`update_endpoint.py NAME --instance-pools A,B,C --yes` replaces the fleet
blue/green (new fleet up, then cutover); quota for both fleets is needed briefly.
