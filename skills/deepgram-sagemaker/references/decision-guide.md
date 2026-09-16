# Decision guide: which Deepgram listing, which mode, which instance

Ask only the questions whose answers change the outcome. Most customers need
three or four of these.

## 1. Speech-to-text or text-to-speech?

| Need | Family | Listings |
|---|---|---|
| Transcribe speech (calls, meetings, media, dictation) | **Nova-3** | `nova-3-mono-streaming`, `nova-3-mono-batch`, `nova-3-multi-streaming`, `nova-3-multi-batch` |
| Voice agent / conversational turn-taking (needs end-of-turn detection, low latency) | **Flux** | `flux-en`, `flux-multi` |
| Synthesize speech (IVR prompts, agent replies, narration) | **Aura-2** | `aura-2` |
| Turn-based synthesis for voice agents on `/v2/speak` (interruptible turns) | **Flux TTS** | `flux-tts` |

## 2. Live audio or pre-recorded files? (STT only)

- **Live** (microphone, phone call, meeting in progress) → a **Streaming**
  listing. Invoked with `InvokeEndpointWithBidirectionalStream` (HTTP/2, port
  8443). Max 30 minutes per connection; reconnect for longer sessions.
- **Pre-recorded files** → a **Batch** listing, invoked with `InvokeEndpoint`
  (**sync**, ≤ 25 MB per request, real-time endpoint). Longer files: split them,
  or stream them at faster than real time through a Streaming listing.
- **Asynchronous endpoints** (`InvokeEndpointAsync`, S3 in/out, up to 1 GB,
  scale-to-zero) are **temporarily not supported for Marketplace-hosted
  Deepgram**. If the customer's use case needs them, they should reach out to a
  Deepgram representative; do not deploy one.
- Flux is streaming only. Aura-2 and Flux TTS serve both streaming and sync
  from one endpoint.

## 3. One known language, or mixed / unknown? (Nova-3 and Flux)

- One language per request, known up front → **Monolingual** listing. Each
  *version* of the listing carries a set of languages (read the version title,
  e.g. `General + Medical (de/en/es/fr/hi/ja)` vs `General (zh/ja/ko/vi/id/th)`).
  Deploy the version that has their language; the request sends `language=<code>`.
- Speakers switch languages, or the language is unknown → **Multilingual**
  listing. Request sends `language=multi` (Nova-3) or `model=flux-general-multi` (Flux).

## 4. Which region?

Wherever the audio comes from / their other AWS resources are. Constraints:
- The model package ARN is per-region; resolve it for that region.
- GPU quota is per-region; check it there.
- FIPS endpoints exist only in US East (N. Virginia, Ohio), US West (N.
  California, Oregon), Canada (Central) and GovCloud. Deepgram is not listed in
  GovCloud Marketplace.
- If capacity for a type is short in one region, another region is often the
  fastest fix.

## 5. How much concurrency?

**Do not size from a table, and do not quote per-instance numbers.** Capacity
per instance varies several-fold with the request features (interim results,
smart formatting, diarization, keyterms, multilingual) and with the audio
itself, so any generic figure would be wrong for the customer's actual traffic.

- **Recommend the customer measure it themselves.** Deploy one instance, ramp
  concurrent streams with their real request parameters until latency (or
  `ConcurrentRequestsPerModel` vs first-chunk latency in CloudWatch) degrades,
  then set `--instance-count` and the auto-scaling `--target` to 70–80 % of
  that level. The load drivers in this repository exist for exactly this:
  `python-stt/stt_wav_stress.py stream --connections N`,
  `python-flux/flux_stress.py`, `python-tts/tts_stress.py`,
  `python-flux-tts/`.
- **For a planning estimate before testing, ask a Deepgram representative**,
  who can size against Deepgram's benchmarks for the specific product, GPU and
  feature set. Do not invent or extrapolate a number.

## 6. Instance type

Take the `recommended` type from `products.json` → `instance_profiles`, confirm it
appears in the package's `supported_realtime_instance_types` (printed by
`resolve_model_package_arn.py`) and that `check_quota.py` shows room. Only then
present alternatives:

- **Nova-3**: any single-GPU `*.2xlarge` in g6 / g6e / g5 / g4dn / g7 / g7e.
- **Flux**: same minus **g4dn** (no T4 kernel; produces empty/garbage output).
- **Aura-2**: multi-GPU **`*.12xlarge`** only.
- **Flux TTS**: g6 or newer `*.2xlarge`; **no g5, no g4dn**.

**Deploy with a pool, not one type.** `--instance-pools default` uses the
family's `default_pool` (recommended type first, then similar-capacity newer
generation, older generation last). SageMaker falls back down the list when a
type has no capacity — single-type deploys fail on capacity routinely. Quota
does not fall back: every type in the pool needs quota ≥ 1, and
`deploy_endpoint.py` drops zero-quota rungs. Ordering rules and the single-type
exceptions: `instance-pools.md`.

## 7. Compliance and networking

- FIPS 140-3: pass `--fips` to the scripts; the streaming client keeps port 8443
  on `runtime-fips.sagemaker.<region>.amazonaws.com`.
- Private connectivity: an interface VPC endpoint for `sagemaker.runtime` keeps
  invocations off the public internet. The container itself has no outbound
  network (Marketplace network isolation) and receives no AWS credentials.
- Data: audio is processed in their account; nothing leaves it. Marketplace
  metering rides SageMaker's own channel.

## Output of this phase

One `slug` from `products.json`, a mode (`streaming` / `sync` / `async`), a
region, an ordered instance pool (default: the family's `default_pool`; a single
type only with a stated reason), an instance count, and whether they want
auto-scaling. Then go to Phase 2.
