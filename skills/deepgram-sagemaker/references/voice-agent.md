# Voice Agent on SageMaker: configure and drive it

The `voice-agent` product runs a complete Deepgram Voice Agent inside the
endpoint: Flux speech-to-text, an LLM bundled in the container, and Flux
text-to-speech. Nothing leaves the endpoint, and Marketplace endpoints run
with network isolation, so the agent works with no outbound network access.

The protocol is the hosted [Voice Agent API](https://developers.deepgram.com/docs/voice-agent)
carried over SageMaker's bidirectional stream. Configuration is **per session**,
in the `Settings` message the client sends first. The endpoint itself has
nothing to configure: deploy it once and every application or caller can use
its own prompt, voice and functions.

## What the endpoint serves

| Path | Transport | Use |
|---|---|---|
| `v1/agent/converse` | bidirectional stream (port 8443) | the Voice Agent: caller audio in, agent audio and JSON events out |
| `v2/listen` | bidirectional stream | standalone Flux speech-to-text (same as the `flux-multi` product) |
| `v2/speak` | bidirectional stream or `InvokeEndpoint` | standalone Flux text-to-speech (same as the `flux-tts` product) |

The LLM is only reachable through an agent session: there is no standalone
LLM endpoint (`v1/think` returns 404), and Conversational Mode runs only
inside `v1/agent/converse`. Requests to paths the endpoint does not serve
return 404 or 405 and are not billed.

There is no synchronous agent call. Leave `ModelQueryString` out entirely for
`v1/agent/converse`: SageMaker rejects an empty string (HTTP 400
`ValidationError`), and any query parameter the endpoint does not recognise
refuses the session at start (HTTP 424 to the client).

## Deploying

- **Instance:** four GPUs on one host. `ml.g7.24xlarge` is recommended, then
  `ml.g7e.24xlarge` and `ml.g6e.12xlarge` (`instance_profiles.voice-agent-4-gpu`
  in `products.json`). Quota for these large types is often 0 by default:
  run `check_quota.py` first.
- **Timeouts:** the container loads three models and the LLM before `/ping`
  passes, about 6 minutes on `ml.g7.24xlarge`. `deploy_endpoint.py` sets
  `ModelDataDownloadTimeoutInSeconds=1800` and
  `ContainerStartupHealthCheckTimeoutInSeconds=2400` for this product. If you
  deploy by hand, set both: SageMaker's own defaults stop the endpoint before
  the container is ready.
- **Expect** 10 to 15 minutes from `CreateEndpoint` to InService.

## The Settings message

Send it as a text frame (`DataType: "UTF8"`) right after the stream opens,
before any audio. The minimal working shape, and what `invoke_test.py` sends:

```json
{
  "type": "Settings",
  "audio": {
    "input":  { "encoding": "linear16", "sample_rate": 16000 },
    "output": { "encoding": "linear16", "sample_rate": 24000, "container": "none" }
  },
  "agent": {
    "greeting": "Hi, thanks for calling. How can I help?",
    "listen": { "provider": { "type": "deepgram", "model": "flux-general-multi", "version": "v2" } },
    "think": {
      "provider": { "type": "deepgram", "model": "phonellm-mini-alpha-2-fp8" },
      "prompt": "You are a friendly restaurant host on a phone call. Keep every reply to one or two short spoken sentences."
    },
    "speak": { "provider": { "type": "deepgram", "version": "v2", "model": "flux-alexis-en" } }
  }
}
```

Field by field, as it applies on SageMaker:

| Field | On SageMaker |
|---|---|
| `audio.input` | The format of the caller audio you stream. `linear16` at 16000 Hz is what the kit tests; match it to what you actually send. |
| `audio.output` | The format of the agent audio you receive. `linear16`, 24000 Hz, `"container": "none"` gives raw PCM frames. |
| `agent.greeting` | Optional. Spoken as soon as the session starts. |
| `agent.listen.provider` | `type: deepgram`, `version: v2`, `model: flux-general-multi` (the speech-to-text model in the container). |
| `agent.think.provider` | `type: deepgram`, `model: phonellm-mini-alpha-2-fp8`: the LLM bundled in the container, selected by name exactly as you select a Deepgram LLM on the hosted API. No `endpoint`, no API key. Other providers and models need outbound network access and are not available. |
| `agent.think.prompt` | Your system prompt. Keep spoken replies short: reply length drives both latency and per-session cost. |
| `agent.think.functions` | Client-side functions (below). |
| `agent.speak.provider` | `type: deepgram`, `version: v2`, `model: flux-<voice>-en`: a Flux TTS English voice from the [TTS models page](https://developers.deepgram.com/docs/tts-models-languages-overview). |
| `agent.think_live` | Optional. `{"provider": {"type": "deepgram", "version": "v1"}}` turns on **Conversational Mode**: backchannels ("mm-hm") and more natural turn-taking. |

Any `endpoint` URL in Settings (think, speak, or a function) must be a public
`https`/`wss` URL, as on the hosted API. With network isolation, which
Marketplace endpoints always have, such URLs are unreachable, so keep every
provider on the bundled models and run functions client-side.

Full field reference: [Configure the Voice Agent](https://developers.deepgram.com/docs/configure-voice-agent).

## During the session

**Keep sending audio for the whole call**, silence included. End-of-turn is
detected from the audio itself: a client that stops sending when nobody talks
never ends the caller's turn, and the agent never answers. Send 20 to 100 ms
binary frames at real time. `{"type": "KeepAlive"}` keeps an idle session
open while you are deliberately not sending audio.

Messages the client sends (text frames):

| Message | Effect |
|---|---|
| `Settings` | First message; configures the session. |
| `UpdatePrompt` `{"prompt": "…"}` | Changes the agent's instructions mid-call; answered by `PromptUpdated`. |
| `InjectAgentMessage` `{"message": "…"}` | Makes the agent say this now. |
| `InjectUserMessage` `{"content": "…"}` | A typed user turn; the agent replies to it. |
| `FunctionCallResponse` | Your answer to a `FunctionCallRequest`. |
| `KeepAlive` | Keeps the session open without audio. |

Events the endpoint sends (text frames; agent audio arrives as binary frames)
include `Welcome`, `SettingsApplied`, `ConversationText` (`role` user or
assistant), `UserStartedSpeaking` (stop playing agent audio: this is barge-in),
`AgentAudioDone`, `FunctionCallRequest`, `PromptUpdated`, `Warning` and `Error`,
as on the hosted API ([full list](https://developers.deepgram.com/docs/voice-agent)).
`History` and `LatencyReport` describe the session; a client that does not use
them can ignore them.

### Client-side functions

Declare them in `agent.think.functions` without an `endpoint`:

```json
"functions": [{
  "name": "book_table",
  "description": "Book a restaurant table.",
  "parameters": {"type": "object",
                 "properties": {"party_size": {"type": "integer"}, "time": {"type": "string"}},
                 "required": ["party_size", "time"]}
}]
```

The agent sends `FunctionCallRequest` with `functions[]` entries carrying `id`,
`name`, `arguments` (a JSON string) and `client_side: true`. Answer with
`{"type": "FunctionCallResponse", "id": "<id>", "name": "book_table", "content": "<result text>"}`
and the agent speaks a follow-up. The LLM decides whether to call a function;
a prompt that says when to use it makes calls more reliable.

## Testing your configuration

```bash
uv run scripts/invoke_test.py <endpoint> --region <region> --product voice-agent
uv run scripts/invoke_test.py <endpoint> --region <region> --product voice-agent --think-live
uv run scripts/invoke_test.py <endpoint> --region <region> --product voice-agent --settings my-settings.json
```

PASS means the session applied the Settings, spoke an audible greeting (if one
is set), heard 6 seconds of the bundled sample as the caller, and spoke an
audible reply. `--json` prints the exact request, including the Settings sent.
The full example client and a 15-scenario end-to-end suite (barge-in,
functions, injections, keep-alive, concurrency) are in this repository's
`python-voice-agent/`.

## Billing

A Voice Agent session is billed per second of connection time, from stream
open to close. The speech-to-text and text-to-speech the agent uses inside the
session are included. Close sessions when the call ends: an open, idle session
is still billed. Standalone `v2/listen` and `v2/speak` requests on the same
endpoint are billed like the Flux STT and Flux TTS products. The rate is on the
listing.

## Capacity

Sessions per instance vary several-fold with prompt length, reply length and
whether Conversational Mode is on. Measure on your own endpoint with your own
Settings (`python-voice-agent/` has a concurrent scenario to start from) and ask
a Deepgram representative for a planning estimate.
