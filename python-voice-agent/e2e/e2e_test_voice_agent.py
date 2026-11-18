#!/usr/bin/env python3
"""Voice Agent e2e battery — `/v1/agent/converse` (plus the standalone `/v2/listen`
and `/v2/speak` the same endpoint serves) via SageMaker.

    AWS_PROFILE=my-profile uv run e2e/e2e_test_voice_agent.py <endpoint>
    uv run e2e/e2e_test_voice_agent.py <endpoint> --list
    ... <endpoint> --scenario single_turn_reply --scenario barge_in

Caller audio is synthesized with the endpoint's own text-to-speech, so the run
needs no fixtures and no second endpoint. Assertions are on protocol events,
the presence and loudness of agent audio, and latency bounds — never on the
LLM's exact wording, which varies run to run.

Judge by the `PASSED:` line. The bidirectional-streaming transport reports a
teardown error on nearly every connection; that alone is not a failure.

  greeting_audio            Welcome, SettingsApplied, and an audible greeting.
  single_turn_reply         caller speaks → assistant ConversationText + audible
                            reply; speech-end → first reply audio reported, FAIL
                            only above REPLY_LATENCY_BOUND_S.
  multi_turn_3              three caller turns, three replies, one session.
  think_live_turn           same as single_turn_reply with Conversational Mode on.
  barge_in                  caller talks over a long agent line → UserStartedSpeaking
                            and the agent's audio stops within BARGE_IN_STOP_S.
  inject_agent_message      InjectAgentMessage is spoken (text echoed + audio).
  inject_user_message       InjectUserMessage (a typed turn) gets a spoken reply.
  update_prompt             UpdatePrompt is acknowledged with PromptUpdated.
  function_call_client_side the agent calls a client-side function; the client
                            answers with FunctionCallResponse and the agent
                            speaks a follow-up. SKIP if the model chose not to
                            call it (a model decision, not an endpoint fault).
  keepalive_idle_20s        20 s with no audio, only KeepAlive: the session stays
                            open and still answers afterwards.
  invalid_settings_error    a malformed Settings message → Error event, and the
                            session ends instead of hanging.
  unknown_query_param_rejected  NEGATIVE control: an unrecognised query parameter
                            is refused at session start, not ignored.
  concurrent_4              four simultaneous sessions each get a reply.
  standalone_stt_v2_listen  `/v2/listen` (Flux) on the same endpoint transcribes
                            the caller line.
  standalone_tts_v2_speak   `/v2/speak` (batch) on the same endpoint returns
                            audible audio.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from e2e.e2e_test_common import (  # noqa: E402
    CALLER_KEYWORDS,
    CALLER_LINES,
    DEFAULT_VOICE,
    GREETING,
    INPUT_SAMPLE_RATE,
    INTERRUPT_LINE,
    LONG_AGENT_LINE,
    OUTPUT_SAMPLE_RATE,
    print_summary_table,
    select_scenarios,
    spoken_verdict,
)
from voice_agent_client import (  # noqa: E402
    AgentSession,
    PerStreamClient,
    quiet_completed_stream_errors,
    agent_settings,
    bidi_endpoint_uri,
    ensure_env_credentials,
    runtime_client,
    synthesize,
    transcribe,
)

logger = logging.getLogger("voice-agent-e2e")

# Generous on purpose: this gates "the agent answered in a conversational time",
# not a performance target. The measured number is printed in the notes.
REPLY_LATENCY_BOUND_S = 5.0
# After the caller starts talking over the agent, agent audio already in flight
# may still arrive for a moment (the server sends slightly ahead of real time).
BARGE_IN_STOP_S = 2.0
SCENARIO_TIMEOUT_S = 240.0


class Ctx:
    """What every scenario needs: endpoint, clients, and cached caller audio."""

    def __init__(self, args):
        self.endpoint = args.endpoint
        self.voice = args.voice
        self.region = args.region
        self.fips = args.fips
        self.rt = runtime_client(args.region, args.fips)
        self._pcm: dict[str, bytes] = {}

    def client(self) -> PerStreamClient:
        return PerStreamClient(self.region, self.fips)

    def caller(self, text: str) -> bytes:
        if text not in self._pcm:
            self._pcm[text] = synthesize(self.rt, self.endpoint, text, self.voice)
        return self._pcm[text]


def row(name: str, ok: bool, notes: str, t0: float, *, skipped: bool = False) -> dict:
    return {"scenario": name, "ok": ok, "skipped": skipped, "notes": notes,
            "elapsed_s": time.monotonic() - t0}


async def open_session(ctx: Ctx, client, *, greeting: bool = True, think_live: bool = False,
                       functions: list[dict] | None = None, prompt: str | None = None,
                       cid: int = 0) -> tuple[AgentSession, str]:
    """Open a session, send Settings, wait for SettingsApplied (and the greeting).

    Returns `(session, problem)`; `problem` is "" when the session is ready.
    """
    s = AgentSession(client, ctx.endpoint, cid=cid)
    if not await s.start():
        return s, s.errors[0]
    t = time.monotonic()
    kw = {"prompt": prompt} if prompt else {}
    await s.send_json(agent_settings(voice=ctx.voice, greeting=GREETING if greeting else None,
                                     think_live=think_live, functions=functions, **kw))
    if not await s.wait(lambda: s.first("SettingsApplied") or s.errors, 30):
        return s, "no SettingsApplied within 30 s"
    if s.errors:
        return s, f"Error after Settings: {s.errors[0]}"
    if greeting and not await s.wait_reply(t, timeout=60):
        return s, "no greeting (AgentAudioDone) within 60 s"
    return s, ""


async def caller_turn(ctx: Ctx, s: AgentSession, text: str, timeout: float = 45.0) -> dict:
    """Speak `text`, wait for the reply. Returns a dict describing the turn."""
    mark = time.monotonic()
    speech_end = await s.speak(ctx.caller(text))
    replied = await s.wait_reply(mark, timeout)
    first = s.first_audio_after(speech_end)
    reply_pcm = s.audio_between(speech_end)
    ok_audio, audio_note, stats = spoken_verdict(reply_pcm)
    assistant = s.texts("assistant", mark)
    user = s.texts("user", mark)
    latency = (first - speech_end) if first else None
    problems = []
    if not replied:
        problems.append("no reply (AgentAudioDone) in time")
    if not ok_audio:
        problems.append(f"reply audio: {audio_note}")
    if not assistant:
        problems.append("no assistant ConversationText")
    if s.errors:
        problems.append(f"Error: {s.errors[0][:80]}")
    if latency is not None and latency > REPLY_LATENCY_BOUND_S:
        problems.append(f"reply latency {latency:.2f}s > {REPLY_LATENCY_BOUND_S}s")
    return {"ok": not problems, "problems": problems, "latency": latency,
            "reply_s": stats.get("duration_s", 0.0), "assistant": assistant, "user": user}


def describe(turn: dict) -> str:
    lat = f"{turn['latency'] * 1000:.0f}ms" if turn["latency"] is not None else "n/a"
    return f"latency={lat} reply={turn['reply_s']:.1f}s"


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

async def sc_greeting_audio(ctx):
    t0 = time.monotonic()
    client = ctx.client()
    s = None
    try:
        s, problem = await open_session(ctx, client)
        if problem:
            return row("greeting_audio", False, problem, t0)
        if not s.first("Welcome"):
            return row("greeting_audio", False, "no Welcome", t0)
        ok, note, stats = spoken_verdict(s.audio_between(0))
        return row("greeting_audio", ok,
                   note or f"greeting {stats['duration_s']:.1f}s rms={stats['rms']:.0f}", t0)
    finally:
        if s:
            await s.finish()
        await client.close()


async def _one_turn_scenario(ctx, name: str, think_live: bool):
    t0 = time.monotonic()
    client = ctx.client()
    s = None
    try:
        s, problem = await open_session(ctx, client, think_live=think_live)
        if problem:
            return row(name, False, problem, t0)
        turn = await caller_turn(ctx, s, CALLER_LINES[0])
        notes = describe(turn) if turn["ok"] else "; ".join(turn["problems"])
        return row(name, turn["ok"], notes, t0)
    finally:
        if s:
            await s.finish()
        await client.close()


async def sc_single_turn_reply(ctx):
    return await _one_turn_scenario(ctx, "single_turn_reply", think_live=False)


async def sc_think_live_turn(ctx):
    return await _one_turn_scenario(ctx, "think_live_turn", think_live=True)


async def sc_multi_turn_3(ctx):
    t0 = time.monotonic()
    client = ctx.client()
    s = None
    try:
        s, problem = await open_session(ctx, client)
        if problem:
            return row("multi_turn_3", False, problem, t0)
        notes = []
        for i, line in enumerate(CALLER_LINES):
            turn = await caller_turn(ctx, s, line)
            if not turn["ok"]:
                return row("multi_turn_3", False, f"turn {i + 1}: " + "; ".join(turn["problems"]), t0)
            notes.append(describe(turn))
        return row("multi_turn_3", True, " | ".join(notes), t0)
    finally:
        if s:
            await s.finish()
        await client.close()


async def sc_barge_in(ctx):
    """Talk over a long agent line; the agent must notice and stop talking."""
    t0 = time.monotonic()
    client = ctx.client()
    s = None
    try:
        s, problem = await open_session(ctx, client)
        if problem:
            return row("barge_in", False, problem, t0)
        mark = time.monotonic()
        await s.send_json({"type": "InjectAgentMessage", "message": LONG_AGENT_LINE})
        if not await s.wait(lambda: s.first_audio_after(mark) is not None, 20):
            return row("barge_in", False, "long agent line produced no audio", t0)
        # Let the agent get a couple of seconds in, then interrupt.
        await s.wait(lambda: False, 2.5)
        barge_at = time.monotonic()
        await s.speak(ctx.caller(INTERRUPT_LINE))
        hit = s.first("UserStartedSpeaking", barge_at)
        if not hit:
            await s.wait(lambda: s.first("UserStartedSpeaking", barge_at) is not None, 5)
            hit = s.first("UserStartedSpeaking", barge_at)
        if not hit:
            return row("barge_in", False, "no UserStartedSpeaking while the caller talked over the agent", t0)
        heard_at = hit[0]
        # Audio of the interrupted line that still arrived after the grace period,
        # before the agent's next statement began.
        nxt = s.first("ConversationText", heard_at + 0.01, role="assistant")
        window_end = nxt[0] if nxt else time.monotonic()
        late = s.audio_between(heard_at + BARGE_IN_STOP_S, window_end)
        late_s = len(late) / 2 / OUTPUT_SAMPLE_RATE
        ok = late_s < 0.5
        notes = (f"UserStartedSpeaking {heard_at - barge_at:.2f}s after barge-in; "
                 f"{late_s:.2f}s of agent audio after +{BARGE_IN_STOP_S:.0f}s")
        return row("barge_in", ok, notes if ok else "agent kept talking: " + notes, t0)
    finally:
        if s:
            await s.finish()
        await client.close()


async def sc_inject_agent_message(ctx):
    t0 = time.monotonic()
    client = ctx.client()
    s = None
    line = "Thanks for waiting, your table is confirmed."
    try:
        s, problem = await open_session(ctx, client)
        if problem:
            return row("inject_agent_message", False, problem, t0)
        mark = time.monotonic()
        await s.send_json({"type": "InjectAgentMessage", "message": line})
        await s.wait_reply(mark, timeout=30)
        said = s.texts("assistant", mark)
        ok_audio, note, stats = spoken_verdict(s.audio_between(mark))
        norm = lambda x: re.sub(r"[^a-z ]", "", x.lower())  # noqa: E731
        echoed = any(norm("your table is confirmed") in norm(t) for t in said)
        problems = []
        if not echoed:
            problems.append(f"injected text not echoed (got {said[:1]})")
        if not ok_audio:
            problems.append(note)
        if s.errors:
            problems.append(f"Error: {s.errors[0][:80]}")
        return row("inject_agent_message", not problems,
                   "; ".join(problems) or f"spoken {stats['duration_s']:.1f}s", t0)
    finally:
        if s:
            await s.finish()
        await client.close()


async def sc_inject_user_message(ctx):
    t0 = time.monotonic()
    client = ctx.client()
    s = None
    try:
        s, problem = await open_session(ctx, client)
        if problem:
            return row("inject_user_message", False, problem, t0)
        mark = time.monotonic()
        await s.send_json({"type": "InjectUserMessage", "content": CALLER_LINES[0]})
        replied = await s.wait_reply(mark, timeout=45)
        ok_audio, note, stats = spoken_verdict(s.audio_between(mark))
        problems = []
        if not replied:
            problems.append("no reply")
        if not s.texts("assistant", mark):
            problems.append("no assistant ConversationText")
        if not ok_audio:
            problems.append(note)
        if s.errors:
            problems.append(f"Error: {s.errors[0][:80]}")
        return row("inject_user_message", not problems,
                   "; ".join(problems) or f"reply {stats['duration_s']:.1f}s", t0)
    finally:
        if s:
            await s.finish()
        await client.close()


async def sc_update_prompt(ctx):
    t0 = time.monotonic()
    client = ctx.client()
    s = None
    try:
        s, problem = await open_session(ctx, client, greeting=False)
        if problem:
            return row("update_prompt", False, problem, t0)
        mark = time.monotonic()
        await s.send_json({"type": "UpdatePrompt",
                           "prompt": "If asked about parking, say there is a garage next door."})
        got = await s.wait(lambda: s.first("PromptUpdated", mark) or s.errors, 15)
        if s.errors:
            return row("update_prompt", False, f"Error: {s.errors[0][:80]}", t0)
        return row("update_prompt", bool(got and s.first("PromptUpdated", mark)),
                   "PromptUpdated" if got else "no PromptUpdated within 15 s", t0)
    finally:
        if s:
            await s.finish()
        await client.close()


AVAILABILITY_FN = {
    "name": "check_availability",
    "description": "Check whether a table is free for a party size at a time. "
                   "Always call this before answering any question about availability.",
    "parameters": {
        "type": "object",
        "properties": {
            "party_size": {"type": "integer", "description": "number of guests"},
            "time": {"type": "string", "description": "requested date and time"},
        },
        "required": ["party_size", "time"],
    },
}


async def sc_function_call_client_side(ctx):
    t0 = time.monotonic()
    client = ctx.client()
    s = None
    try:
        s, problem = await open_session(
            ctx, client, greeting=False, functions=[AVAILABILITY_FN],
            prompt="You are a restaurant host. To answer any availability question you MUST "
                   "call check_availability first. Keep replies to one short sentence.")
        if problem:
            return row("function_call_client_side", False, problem, t0)
        mark = time.monotonic()
        await s.send_json({"type": "InjectUserMessage", "content": CALLER_LINES[0]})
        await s.wait(lambda: s.first("FunctionCallRequest", mark) or s.first("AgentAudioDone", mark), 45)
        req = s.first("FunctionCallRequest", mark)
        if not req:
            if s.errors:
                return row("function_call_client_side", False, f"Error: {s.errors[0][:80]}", t0)
            return row("function_call_client_side", True,
                       "model answered without calling the function (a model decision)", t0, skipped=True)
        calls = [f for f in req[1].get("functions", []) if f.get("name") == AVAILABILITY_FN["name"]]
        if not calls:
            return row("function_call_client_side", False, f"unexpected request: {str(req[1])[:80]}", t0)
        call = calls[0]
        if not call.get("client_side", True):
            return row("function_call_client_side", False, "function was not marked client_side", t0)
        answered = time.monotonic()
        await s.send_json({"type": "FunctionCallResponse", "id": call.get("id", ""),
                           "name": call["name"],
                           "content": json.dumps({"available": True, "time": "7:00 PM"})})
        await s.wait_reply(answered, timeout=45)
        ok_audio, note, stats = spoken_verdict(s.audio_between(answered))
        problems = []
        if not ok_audio:
            problems.append(f"no spoken follow-up: {note}")
        if s.errors:
            problems.append(f"Error: {s.errors[0][:80]}")
        return row("function_call_client_side", not problems,
                   "; ".join(problems) or f"args={call.get('arguments', '')[:40]} follow-up {stats['duration_s']:.1f}s",
                   t0)
    finally:
        if s:
            await s.finish()
        await client.close()


async def sc_keepalive_idle_20s(ctx):
    t0 = time.monotonic()
    client = ctx.client()
    s = None
    try:
        s, problem = await open_session(ctx, client)
        if problem:
            return row("keepalive_idle_20s", False, problem, t0)
        for _ in range(4):                     # 20 s: no audio, KeepAlive every 5 s
            if s.closed:
                break
            await s.send_json({"type": "KeepAlive"})
            await s.wait(lambda: False, 5, feed_silence=False)
        if s.closed:
            return row("keepalive_idle_20s", False, "session closed during the idle period", t0)
        if s.errors:
            return row("keepalive_idle_20s", False, f"Error: {s.errors[0][:80]}", t0)
        mark = time.monotonic()
        await s.send_json({"type": "InjectAgentMessage", "message": "Are you still there?"})
        await s.wait_reply(mark, timeout=30)
        ok, note, _ = spoken_verdict(s.audio_between(mark))
        return row("keepalive_idle_20s", ok,
                   "still open and answering after 20 s idle" if ok else f"no audio after idle: {note}", t0)
    finally:
        if s:
            await s.finish()
        await client.close()


async def sc_invalid_settings_error(ctx):
    t0 = time.monotonic()
    client = ctx.client()
    s = AgentSession(client, ctx.endpoint)
    try:
        if not await s.start():
            return row("invalid_settings_error", False, s.errors[0], t0)
        bad = agent_settings(voice=ctx.voice)
        bad["audio"]["input"]["sample_rate"] = "sixteen thousand"   # wrong type
        await s.send_json(bad)
        await s.wait(lambda: bool(s.errors) or s.closed, 20)
        if not s.errors:
            return row("invalid_settings_error", False,
                       "malformed Settings produced no Error" + (" (session closed)" if s.closed else ""), t0)
        await s.wait(lambda: s.closed, 15, feed_silence=False)
        notes = f"Error: {s.errors[0][:60]}"
        if not s.closed:
            return row("invalid_settings_error", False, notes + "; session did not end", t0)
        return row("invalid_settings_error", True, notes + "; session ended", t0)
    finally:
        await s.finish()
        await client.close()


async def sc_unknown_query_param_rejected(ctx):
    """NEGATIVE control: a parameter the endpoint does not recognise must be
    refused at session start. Passing means the session was refused; failing
    means it was silently accepted."""
    t0 = time.monotonic()
    client = ctx.client()
    s = AgentSession(client, ctx.endpoint, query="definitely_not_a_param=1")
    try:
        started = await s.start()
        if started:
            await s.wait(lambda: s.closed or bool(s.errors) or bool(s.first("Welcome")), 10,
                         feed_silence=False)
        refused = not started or bool(s.errors) or (s.closed and not s.first("Welcome"))
        detail = s.errors[0][:70] if s.errors else ("closed before Welcome" if s.closed else "")
        return row("unknown_query_param_rejected", refused,
                   f"refused as expected ({detail})" if refused
                   else "UNKNOWN PARAMETER WAS ACCEPTED", t0)
    finally:
        await s.finish()
        await client.close()


async def sc_concurrent_4(ctx):
    t0 = time.monotonic()
    ctx.caller(CALLER_LINES[0])                 # synthesize once, before the fan-out

    async def one(i: int) -> tuple[bool, str]:
        client = ctx.client()
        s = None
        try:
            s, problem = await open_session(ctx, client, cid=i)
            if problem:
                return False, f"c{i}: {problem}"
            turn = await caller_turn(ctx, s, CALLER_LINES[0])
            return turn["ok"], f"c{i}: " + (describe(turn) if turn["ok"] else "; ".join(turn["problems"]))
        finally:
            if s:
                await s.finish()
            await client.close()

    results = await asyncio.gather(*(one(i) for i in range(4)), return_exceptions=True)
    good = sum(1 for r in results if not isinstance(r, BaseException) and r[0])
    notes = [f"{type(r).__name__}" if isinstance(r, BaseException) else r[1] for r in results]
    bad = [n for r, n in zip(results, notes) if isinstance(r, BaseException) or not r[0]]
    return row("concurrent_4", good == 4, f"{good}/4 ok" + ("; " + "; ".join(bad[:2]) if bad else ""), t0)


async def sc_standalone_stt_v2_listen(ctx):
    t0 = time.monotonic()
    client = ctx.client()
    try:
        turns, errors = await transcribe(client, ctx.endpoint, ctx.caller(CALLER_LINES[0]))
    finally:
        await client.close()
    text = " ".join(turns).lower()
    hits = {w for w in CALLER_KEYWORDS if w in text}
    if errors:
        return row("standalone_stt_v2_listen", False, f"Error: {errors[0][:80]}", t0)
    ok = len(hits) >= 3
    return row("standalone_stt_v2_listen", ok,
               f"{len(hits)}/{len(CALLER_KEYWORDS)} keywords: {text[:60]!r}", t0)


async def sc_standalone_tts_v2_speak(ctx):
    t0 = time.monotonic()
    pcm = await asyncio.to_thread(synthesize, ctx.rt, ctx.endpoint,
                                  "Your table for four is booked for seven tomorrow evening.",
                                  ctx.voice, OUTPUT_SAMPLE_RATE)
    ok, note, stats = spoken_verdict(pcm, OUTPUT_SAMPLE_RATE)
    return row("standalone_tts_v2_speak", ok,
               note or f"{stats['duration_s']:.1f}s rms={stats['rms']:.0f}", t0)


SCENARIOS = {
    "greeting_audio": sc_greeting_audio,
    "single_turn_reply": sc_single_turn_reply,
    "multi_turn_3": sc_multi_turn_3,
    "think_live_turn": sc_think_live_turn,
    "barge_in": sc_barge_in,
    "inject_agent_message": sc_inject_agent_message,
    "inject_user_message": sc_inject_user_message,
    "update_prompt": sc_update_prompt,
    "function_call_client_side": sc_function_call_client_side,
    "keepalive_idle_20s": sc_keepalive_idle_20s,
    "invalid_settings_error": sc_invalid_settings_error,
    "unknown_query_param_rejected": sc_unknown_query_param_rejected,
    "concurrent_4": sc_concurrent_4,
    "standalone_stt_v2_listen": sc_standalone_stt_v2_listen,
    "standalone_tts_v2_speak": sc_standalone_tts_v2_speak,
}


async def main_async(args) -> int:
    ensure_env_credentials(args.region)
    print(f"Bidi URL:    {bidi_endpoint_uri(args.region, args.fips)}")
    print(f"FIPS:        {'yes (--fips)' if args.fips else 'no'}")
    print(f"Endpoint:    {args.endpoint}  voice={args.voice}  caller audio {INPUT_SAMPLE_RATE} Hz")
    ctx = Ctx(args)
    rows = []
    for name in select_scenarios(list(SCENARIOS), args.scenario):
        print(f"\n--- {name} ---", flush=True)
        t0 = time.monotonic()
        try:
            r = await asyncio.wait_for(SCENARIOS[name](ctx), SCENARIO_TIMEOUT_S)
        except asyncio.TimeoutError:
            r = row(name, False, f"timed out after {SCENARIO_TIMEOUT_S:.0f}s", t0)
        except Exception as e:  # noqa: BLE001
            logger.exception("scenario %s raised", name)
            r = row(name, False, f"raised {type(e).__name__}: {e}"[:120], t0)
        status = "SKIP" if r.get("skipped") else ("PASS" if r["ok"] else "FAIL")
        print(f"{status} {name}: {r['notes']}", flush=True)
        rows.append(r)
    print()
    _, failed = print_summary_table(rows)
    return 1 if failed else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("endpoint", nargs="?", help="SageMaker endpoint name")
    p.add_argument("--region", default="us-east-2")
    p.add_argument("--voice", default=DEFAULT_VOICE,
                   help=f"agent voice, also used to synthesize the caller (default {DEFAULT_VOICE})")
    p.add_argument("--scenario", action="append",
                   help="run only this scenario (repeatable, or comma-separated)")
    p.add_argument("--fips", action="store_true",
                   help="Route AWS calls to the FIPS 140-3 endpoints (bidi streaming on "
                        "runtime-fips.sagemaker.<region>.amazonaws.com:8443). OFF by default.")
    p.add_argument("--list", action="store_true", help="list scenarios and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.list:
        for n in SCENARIOS:
            print(n)
        return 0
    if not args.endpoint:
        p.error("endpoint is required unless --list")
    async def run():
        quiet_completed_stream_errors(asyncio.get_running_loop())
        return await main_async(args)
    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
