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

**Gate this clip on CER, not WER.** `e2e_test_common.cer()` exists for exactly
this reason, and the streaming driver selects it automatically when
`--audio-lang ja` is in effect.

The reason is narrower than "Japanese has no spaces". `normalize_for_wer`
turns punctuation into spaces, so the text does tokenise — at *clause*
boundaries. Measured 2026-09-11 against a real nova-3 `language=ja` transcript
of this file: **3 wrong characters out of 135 scored CER 2.22% and WER
36.36%**, because each "word" is a whole clause and a single bad character
fails all of it. WER is therefore not binary here (an earlier version of this
note said it was, based on a raw `.split()` that the driver never calls) — it
is just wildly pessimistic and not comparable to an English WER number.

The signal that matters most is still cheaper than either metric: `finals > 0`
plus the presence terms. That is what distinguishes "the model transcribes"
from "the endpoint is merely alive", which was the whole gap.
