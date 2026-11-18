# python-voice-agent — Deepgram Voice Agent on SageMaker

Example client and end-to-end tests for the Deepgram **Voice Agent** deployed as
a SageMaker endpoint. The whole speech-to-speech loop runs inside the container,
with no outbound network access: Flux speech-to-text, the LLM bundled in the
container, and Flux text-to-speech.

One endpoint serves three surfaces:

| path | transport | what it is |
|---|---|---|
| `/v1/agent/converse` | bidirectional streaming (WebSocket) | the Voice Agent: caller audio in, agent audio + JSON events out |
| `/v2/listen` | bidirectional streaming (WebSocket) | standalone Flux speech-to-text |
| `/v2/speak` | `invoke_endpoint` (batch) or bidirectional streaming | standalone Flux text-to-speech |

## Deploying

The container loads three models and the LLM before it reports healthy, which
takes up to ~25 minutes on first start. Raise both endpoint-config timeouts or
SageMaker gives up first:

- `ModelDataDownloadTimeoutInSeconds`: `1800`
- `ContainerStartupHealthCheckTimeoutInSeconds`: `2400`

The endpoint needs a four-GPU instance; see the listing for the supported types.

## Billing

A Voice Agent session is billed per second of WebSocket connection time. The
speech-to-text and text-to-speech the agent uses inside a session are included.
Standalone `/v2/listen` and `/v2/speak` requests on the same endpoint are billed
like the Flux STT and Flux TTS products.

## Settings

A session starts with one `Settings` message (sent as a text frame), then binary
audio frames. This is what the scripts here send:

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
      "provider": { "type": "open_ai", "model": "phonellm" },
      "endpoint": { "url": "http://127.0.0.1:3081/v1/chat/completions", "headers": {} },
      "prompt": "You are a friendly restaurant host on a phone call. Keep every reply to one or two short spoken sentences."
    },
    "speak": { "provider": { "type": "deepgram", "version": "v2", "model": "flux-alexis-en" } }
  }
}
```

- `think` points at the LLM bundled in the container. The address is local to the
  container; nothing leaves the endpoint.
- Add `"think_live": {"provider": {"type": "deepgram"}}` under `agent` to turn on
  Conversational Mode (backchannels and more natural turn-taking).
- Client-side functions go in `agent.think.functions`; the agent asks for them
  with `FunctionCallRequest` and the client answers with `FunctionCallResponse`.
- Keep sending audio for the whole call, including silence while nobody is
  talking. End-of-turn is detected from the audio itself: a client that just
  stops sending never ends its turn.

## Setup

```bash
cd python-voice-agent
uv sync
```

## Example client

[`voice_agent_client.py`](voice_agent_client.py) holds one short conversation:
greeting, one caller turn, the agent's reply. It prints the conversation and
saves the agent's audio to a WAV.

```bash
# caller line synthesized with the endpoint's own text-to-speech
AWS_PROFILE=my-profile uv run voice_agent_client.py <endpoint> \
    --say "Hi, do you have a table for four tomorrow at seven?" --out agent.wav

# your own recording (16 kHz mono 16-bit WAV), Conversational Mode on
uv run voice_agent_client.py <endpoint> --wav caller16k.wav --think-live
```

Convert other audio with `ffmpeg -i in.wav -ar 16000 -ac 1 -c:a pcm_s16le out.wav`.
The file also exposes `AgentSession`, `transcribe()` (`/v2/listen`) and
`synthesize()` (`/v2/speak`) for your own code.

## End-to-end tests

[`e2e/e2e_test_voice_agent.py`](e2e/e2e_test_voice_agent.py) runs 15 scenarios
against one endpoint. Caller audio is synthesized with the endpoint's own
text-to-speech, so no fixtures or second endpoint are needed.

```bash
AWS_PROFILE=my-profile uv run e2e/e2e_test_voice_agent.py <endpoint>
uv run e2e/e2e_test_voice_agent.py <endpoint> --list
uv run e2e/e2e_test_voice_agent.py <endpoint> --scenario barge_in --scenario concurrent_4
```

| scenario | passes when |
|---|---|
| `greeting_audio` | `Welcome`, `SettingsApplied`, and an audible greeting |
| `single_turn_reply` | the caller speaks; an assistant `ConversationText` and an audible reply follow within 5 s of the end of speech (the measured latency is printed) |
| `multi_turn_3` | three caller turns in one session each get a reply |
| `think_live_turn` | `single_turn_reply` with Conversational Mode on |
| `barge_in` | the caller talks over a long agent line; `UserStartedSpeaking` arrives and the agent's audio stops within 2 s |
| `inject_agent_message` | `InjectAgentMessage` text is echoed as `ConversationText` and spoken |
| `inject_user_message` | a typed `InjectUserMessage` turn gets a spoken reply |
| `update_prompt` | `UpdatePrompt` is acknowledged with `PromptUpdated` |
| `function_call_client_side` | the agent calls a client-side function, the client answers, the agent speaks a follow-up. **SKIP** if the LLM chose not to call it |
| `keepalive_idle_20s` | 20 s with only `KeepAlive`: the session stays open and still answers |
| `invalid_settings_error` | a malformed `Settings` returns an `Error` and the session ends instead of hanging |
| `unknown_query_param_rejected` | a query parameter the endpoint does not recognise is refused at session start |
| `concurrent_4` | four simultaneous sessions each get a reply |
| `standalone_stt_v2_listen` | `/v2/listen` on the same endpoint transcribes the caller line |
| `standalone_tts_v2_speak` | `/v2/speak` on the same endpoint returns audible audio |

Assertions are on protocol events, audio presence and loudness, and latency
bounds, never on the LLM's exact wording. Judge a run by the final
`PASSED: N  FAILED: N  TOTAL: N` line; the bidirectional-streaming transport
reports a teardown error on nearly every connection, and that alone is not a
failure. `--fips` routes every AWS call to the FIPS 140-3 endpoints.

Capacity depends heavily on prompt length, reply length and whether
Conversational Mode is on. Measure on your own endpoint with your own settings,
and ask a Deepgram representative for a planning estimate.
