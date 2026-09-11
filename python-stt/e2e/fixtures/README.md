# Committed e2e audio fixtures

Unlike `spacewalk.wav`, which `e2e_test_common.download_sample()` fetches from
`dpgr.am` at run time, everything here is **committed on purpose**. A
target-language check that can silently degrade into "no audio available,
skipped" is worse than no check at all.

## `ja_reference_24k.wav`

Mono / 24 kHz / 16-bit / ~22.5 s. RMS 1171, peak 15856 — comfortably above the
150 silence floor the drivers use.

**Why it exists.** Every other clip in the suite is English, which left the
east-asian nova-3 bundles (`zh/ja/ko/vi/id/th`) unverifiable. Those bundles
carry no `en` model, so `--language en` is rejected outright, while English
audio under `--language ja` is *accepted* and returns nothing. Measured
2026-09-11 against the published eastasia streaming version: every connection
lived its full ~37 s and returned `finals=0` — indistinguishable from a model
that is simply broken. That release was signed off on liveness alone, and this
fixture is what makes the next one verifiable.

**Provenance.** Synthesised with Deepgram TTS, so it is ours to commit and is
reproducible:

```bash
curl -X POST "https://api.deepgram.com/v1/speak?model=aura-2-fujin-ja&encoding=linear16&sample_rate=24000" \
  -H "Authorization: Token $DEEPGRAM_API_KEY" -H "Content-Type: application/json" \
  -d '{"text": "<JA_REFERENCE_TEXT from e2e_test_common.py>"}'
```

Two things to know if you regenerate it:

- The response's RIFF header carries a **streaming placeholder size**
  (`0x7fff0024`), so `wave` reports a nonsense frame count until the bytes are
  re-wrapped with correct chunk sizes. The committed file already is.
- `aura-2-fujin-ja` was chosen over `aura-2-izanami-ja` by round-tripping both
  through nova-3 `language=ja`. Both returned confidence 1.0, but izanami
  dropped 宇宙 from 宇宙飛行士 while fujin kept it.

**Verified contract** (re-checked 2026-09-11, nova-3 `language=ja`, confidence
1.0): the transcript contains all of `JA_PRESENCE_TERMS` — 東京, 宇宙飛行士,
午後三時, 記者会見.

**Do not gate this clip on word-level WER.** Japanese is written without
spaces, so `text.split()` yields exactly ONE token for both reference and
hypothesis — measured, not assumed. Any difference at all scores 100% and an
exact match scores 0%, with nothing in between. Judge it on `finals > 0` plus
the presence terms, or add a character-error-rate metric before using a numeric
threshold.
