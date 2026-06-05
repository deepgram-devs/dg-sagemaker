# AGENTS.md

Notes for AI agents working in this repo.

## Keep the top-level README in sync

[`README.md`](README.md) at the repo root is the index of every script, grouped
by service (STT / TTS / Flux) and language (JavaScript / Python / Java). When
you add, rename, remove, or change the purpose of anything user-facing in a
subfolder, update that index in the **same change** — do not leave it for later.
This includes:

- a new script or CLI subcommand (e.g. a new `*_stress.py` mode),
- a new `e2e/` driver or scenario surface,
- a new component/language subfolder (add a section and a pointer to its README),
- a renamed or deleted file that the README links to.

How to keep it consistent:

- Match the existing entry style: a markdown link to the file, then an em-dash
  one-line description at index altitude. Leave the deep detail (flags,
  pass/fail, parameter matrices) in the subfolder's own README and link to it.
- Every path the README links to must exist — verify after editing.
- Mirror the change in the relevant subfolder README too, and if it's an e2e
  driver with a non-trivial runtime, add a row to the wall-clock table below.

The same rule applies to this file: when an e2e driver's runtime, pass/fail
format, or invocation contract changes, update the relevant section here.

## E2E wall-clock expectations

Pick the right runner. The streaming STT e2e includes a real-time
sustained-load scenario whose floor is the duration of the audio file. Long
scenarios will exceed agent harness time limits in many platforms — run the
streaming STT e2e as a detached background process with an explicit long
timeout; the others fit easily in a single agent turn.

| Script | Total wall-clock | Longest scenario | Notes |
|---|---|---|---|
| `python-stt/e2e/e2e_test_streaming.py` | **~17–18 min** | `concurrent_10x_15min` ≈ **915 s** | Plays a ~15-min file at real-time across 10 concurrent WS connections. Do not run inside a short-lived agent subprocess — run as a backgrounded shell command with a 40+ min timeout and poll for the `PASSED:.*FAILED:` line in the tail. |
| `python-stt/e2e/e2e_test_batch.py` | ~1 min | summarize/topics scenarios ~4 s | 22 scenarios, mostly 1–2 s each. Safe to run inline. |
| `python-tts/e2e/e2e_test_batch.py` | ~1.5 min | speed_duration ~12 s | 20 scenarios. Safe inline. |
| `python-tts/e2e/e2e_test_streaming.py` | ~1.5 min | multi_phrase_flush ~17 s | 8 scenarios. Safe inline. |

Wall-clocks measured against single-GPU (STT) and multi-GPU (TTS) SageMaker
endpoints. Network and instance class shift the numbers, but the streaming-STT
floor is set by real-time playback of the 15-min sample file and won't change.

## Invocation defaults

All scripts take the endpoint name as the first positional arg and default
`--region us-east-2`. They use boto3 and respect any standard AWS credential
chain (`AWS_PROFILE`, env vars, instance role). Typical invocation:

```bash
cd python-stt
uv run e2e/e2e_test_streaming.py <endpoint-name>
```

`uv sync` is not required up front — `uv run` resolves the project venv on
first call.

## Pass/fail parsing

The final block is always:

```
=====...
PASSED: N  FAILED: N  TOTAL: N
```

Grep for `^PASSED:` to get the counts; nothing else in the output uses that
prefix. The scenario table immediately above is the per-scenario record.

## Endpoint deletion ordering

If you orchestrate deploy → e2e → cleanup, the cleanup step must wait for the
e2e process to fully exit before calling `delete-endpoint`. The streaming e2e
keeps making invocations until the very end (`concurrent_10x_15min`,
`adversarial_bare_close`). Killing the endpoint mid-run causes the remaining
scenarios to all 5xx and produces a misleading "PASSED: 0" result.
