# ThreadiaVA Voice AI Pipeline - Technical Specification

## Overview

This document defines the technical architecture and implementation specifications for ThreadiaVA's voice AI pipeline. The system enables real-time, natural voice conversations between human callers and an AI agent, with support for backchannels, interruptions, and context preservation.

---

## 1. Infrastructure Layer: LiveKit

### What is LiveKit

LiveKit is an open-source, real-time communication platform built on WebRTC. It provides infrastructure for audio, video, and data streaming with global edge distribution. LiveKit handles SIP termination, room management, and participant routing—abstracting away the complexity of real-time media transport.

ThreadiaVA uses LiveKit Cloud, the managed deployment option, which provides:

- Global edge network with sub-100ms latency to nearest node
- Automatic scaling and orchestration
- SIP integration for telephony
- Agent hosting infrastructure

### SIP Connection Flow

```
Caller (PSTN) → Twilio → SIP → LiveKit Cloud → Room → Agent
```

1. **Inbound Call**: Caller dials Twilio phone number
2. **Twilio Webhook**: Twilio hits your webhook, receives TwiML response
3. **SIP Dial**: TwiML instructs Twilio to `<Dial><Sip>` to LiveKit endpoint
4. **Automatic Answer**: LiveKit answers the SIP call automatically
5. **Handshake**: SIP handshake completes without custom code
6. **RTP Flow**: Audio packets begin streaming immediately
7. **Room Creation**: LiveKit creates a room, places caller as participant
8. **Agent Dispatch**: LiveKit spins up your agent container, joins it to the room

### TwiML Example

```xml
<Response>
  <Dial>
    <Sip>sip:your-agent-id@sip.livekit.cloud;transport=tls</Sip>
  </Dial>
</Response>
```

### Agent Deployment

The agent is Python or Node.js code using LiveKit's Agents SDK. It runs on LiveKit Cloud infrastructure:

```bash
# Deploy agent to LiveKit Cloud
lk cloud deploy
```

The agent joins rooms as a participant, subscribes to audio tracks, and publishes audio back. LiveKit handles container orchestration, scaling, and lifecycle management.

---

## 2. Signal Layer: Deepgram Flux

### Role

Deepgram is the **single source of truth** for user speech state. We use the **Flux model** - Deepgram's first conversational speech recognition model designed specifically for voice agents.

### Why Flux (Not Nova-2/Nova-3)

| Feature | Nova Models | Flux |
|---------|-------------|------|
| Turn detection | Manual (silence timing) | Native (~260ms latency) |
| End-of-turn events | `utterance_end` only | `StartOfTurn`, `EagerEndOfTurn`, `EndOfTurn`, `TurnResumed` |
| False alarm recovery | Manual re-prompting | `TurnResumed` cancels early calls |
| Backchannel window | Manual calculation | `EagerEndOfTurn` provides 150-250ms window |

### Flux Events

| Event | When It Fires | Your Action |
|-------|---------------|-------------|
| **StartOfTurn** | User begins speaking | Trigger barge-in if agent speaking |
| **EagerEndOfTurn** | Medium confidence (~0.5) | Start speculative LLM call (cancellable) + fire backchannel |
| **TurnResumed** | User continued (false alarm) | Cancel speculative LLM call |
| **EndOfTurn** | High confidence (~0.7) | Confirm/send LLM response |
| **Update** | Ongoing transcription | Accumulate transcript in buffer |

### Key Flux Guarantees

1. `EagerEndOfTurn` always contains a **nonempty transcript**
2. `TurnResumed` can **only follow** a preceding `EagerEndOfTurn`
3. `EndOfTurn` transcript **always matches** the preceding `EagerEndOfTurn` transcript
4. If transcript changes after `EagerEndOfTurn`, a `TurnResumed` event fires first

### Configuration

```python
# Flux configuration parameters
eot_threshold = 0.7           # Confidence for EndOfTurn (0.5-0.9)
eager_eot_threshold = 0.5     # Confidence for EagerEndOfTurn (0.3-0.9)
eot_timeout_ms = 1000         # Force EndOfTurn after this silence
```

**Important**: You must set `eager_eot_threshold` to enable `EagerEndOfTurn` and `TurnResumed` events. By default, Flux only emits `Update`, `StartOfTurn`, and `EndOfTurn`.

### API Endpoint

Flux uses `/v2/listen` (not `/v1/listen`):

```
wss://api.deepgram.com/v2/listen?model=flux-general-en&encoding=linear16&sample_rate=16000&eot_threshold=0.7&eager_eot_threshold=0.5
```

### Legacy Mode (Nova-3)

For backwards compatibility, the system also supports Nova-3 with manual silence detection:
- Uses `is_final`, `speech_final`, and `UtteranceEnd` events
- Requires `SilenceMonitor` for timing-based transitions
- 200ms silence → SHORT_PAUSE, 400ms silence → PROMPTING

---

## 3. State Machine

### States (Flux Event-Driven)

The system operates in one of four states at any time. **All transitions are driven by Flux events**, not manual timers:

```
┌─────────────┐
│  LISTENING  │◄─────────────────────────────────────────────┐
└──────┬──────┘                                              │
       │ EagerEndOfTurn                                      │
       ▼                                                     │
┌─────────────┐                                              │
│ SHORT_PAUSE │──── TurnResumed ─────────────────────────────┤
└──────┬──────┘                                              │
       │ EndOfTurn (confirms turn complete)                  │
       ▼                                                     │
┌─────────────┐                                              │
│  PROMPTING  │──── StartOfTurn (user interrupts) ───────────┤
└──────┬──────┘                                              │
       │ First LLM token received                            │
       ▼                                                     │
┌─────────────────┐                                          │
│ AGENT_SPEAKING  │──── output_complete ─────────────────────┤
└────────┬────────┘                                          │
         │ StartOfTurn (barge-in)                            │
         └───────────────────────────────────────────────────┘
```

### State Transitions by Flux Event

| Flux Event | From State | To State | Action |
|------------|------------|----------|--------|
| `StartOfTurn` | LISTENING | LISTENING | Start transcript accumulation |
| `StartOfTurn` | SHORT_PAUSE | LISTENING | Cancel eager LLM, continue accumulating |
| `StartOfTurn` | PROMPTING | LISTENING | Cancel final LLM, preserve interrupted_context |
| `StartOfTurn` | AGENT_SPEAKING | LISTENING | Stop TTS, capture intended/spoken |
| `EagerEndOfTurn` | LISTENING | SHORT_PAUSE | Fire backchannel, start speculative LLM |
| `TurnResumed` | SHORT_PAUSE | LISTENING | Cancel speculative LLM |
| `EndOfTurn` | SHORT_PAUSE | PROMPTING | Confirm LLM call (reuse eager response if available) |
| `EndOfTurn` | LISTENING | PROMPTING | Send final LLM call |

### State Definitions

#### LISTENING

- **Entry**: `StartOfTurn` event, or transition from other states
- **Behavior**: Accumulate Deepgram transcripts into `current_buffer` via `Update` events
- **Exit**: `EagerEndOfTurn` event fires (Flux detects medium-confidence pause)

#### SHORT_PAUSE

- **Entry**: `EagerEndOfTurn` event (medium confidence ~0.5)
- **Behavior**:
  - Fire backchannel if conditions met
  - Start speculative (eager) LLM call
- **Exit**:
  - `TurnResumed` → LISTENING (cancel eager LLM)
  - `EndOfTurn` → PROMPTING (confirm turn complete)

#### PROMPTING

- **Entry**: `EndOfTurn` event (high confidence ~0.7)
- **Behavior**:
  - If eager LLM already sent, reuse response (per Flux transcript guarantee)
  - Otherwise, send final LLM call
- **Exit**:
  - `StartOfTurn` → LISTENING (cancel LLM, user interrupted)
  - First LLM token received → AGENT_SPEAKING

#### AGENT_SPEAKING

- **Entry**: First LLM token streams to TTS
- **Behavior**: Stream tokens to ElevenLabs, output audio to caller
- **Exit**:
  - Output completes → LISTENING
  - `StartOfTurn` (barge-in) → LISTENING

### The Master Guard

**AGENT_SPEAKING blocks all transitions except `StartOfTurn` (barge-in) and `output_complete`.**

```python
if state == State.AGENT_SPEAKING:
    if trigger == "start_of_turn":
        # Barge-in: stop TTS, capture context, transition
        handle_barge_in()
        return State.LISTENING
    elif trigger == "output_complete":
        return State.LISTENING
    else:
        # Ignore all other events during agent output
        return state
```

This prevents race conditions. Agent Speaking is the lock. Only user speech (`StartOfTurn`) or output completion breaks the lock.

### Eager LLM Optimization

The SHORT_PAUSE state enables speculative LLM calls:

```python
async def _on_eager_end_of_turn(self, turn_info):
    # Transition to SHORT_PAUSE
    await self._sm.transition_to(State.SHORT_PAUSE, trigger="eager_eot")

    # Fire backchannel asynchronously
    if self._should_fire_backchannel():
        asyncio.create_task(self._fire_backchannel())

    # Start speculative LLM call (cancellable)
    self._eager_llm_task = asyncio.create_task(
        self._llm_client.generate_response(...)
    )

async def _on_turn_resumed(self):
    # Cancel speculative call - user continued speaking
    if self._eager_llm_task and not self._eager_llm_task.done():
        self._eager_llm_task.cancel()
    await self._sm.transition_to(State.LISTENING, trigger="turn_resumed")

async def _on_end_of_turn(self, turn_info):
    # Cancel eager if still pending
    self._cancel_eager_llm()

    await self._sm.transition_to(State.PROMPTING, trigger="end_of_turn")

    # Reuse eager response if already sent (Flux guarantees transcript match)
    if self._eager_llm_sent:
        self._reset_turn_state()
        return  # Response already streaming

    # Otherwise send final LLM call
    await self._send_final_llm_call()
```

---

## 4. Buffer Management

### Buffers

| Buffer | Purpose |
|--------|---------|
| `current_buffer` | Accumulates user speech in current turn |
| `interrupted_context` | Stores prompt that was interrupted before Claude responded |
| `agent_response_buffer` | Accumulates tokens as Claude streams them |
| `agent_output_position` | Tracks how much of agent response was actually spoken |

### Buffer Flow: Normal Turn

```
User speaks → current_buffer accumulates
End of turn → current_buffer sent to Claude
Claude responds → agent_response_buffer fills
Output completes → clear all buffers
```

### Buffer Flow: User Interrupts During Prompting

```
User speaks → current_buffer accumulates
End of turn → copy current_buffer to interrupted_context, send to Claude
User speaks again → cancel Claude request
                  → new speech goes to current_buffer
End of turn → send both interrupted_context + current_buffer to Claude
Claude responds → clear interrupted_context after successful response
```

### Buffer Flow: User Interrupts During Agent Speaking

```
Agent speaking → agent_response_buffer has full intended response
              → agent_output_position tracks spoken portion
User interrupts → slice agent_response_buffer at agent_output_position
               → store intended vs spoken for context
               → accumulate new speech to current_buffer
End of turn → send full context (intended, spoken, user input) to Claude
```

---

## 5. Backchannel System

### Purpose

Backchannels ("Mhmm", "I see", "Go on") signal that the AI is actively listening. They fire during short pauses when the user is thinking but not finished speaking.

### Backchannel Pool

Responses are weighted by transcript length:

```python
word_count = len(current_buffer.split())

if word_count < 10:
    # User just started
    pool = ["I'm listening", "Go on", "Okay"]

elif word_count < 30:
    # Mid-conversation
    pool = ["Mhmm", "Uh huh", "Okay", "Right"]

else:
    # User has said a lot
    pool = ["Mhmm", "I see", "Right", "Uh huh"]
```

### Backchannel Conditions

All must be true to fire a backchannel:

1. State is SHORT_PAUSE
2. ElevenLabs output layer is idle
3. `current_buffer` has sufficient content (user said something substantive)
4. Cooldown period has passed (3-4 seconds since last backchannel)

```python
def should_fire_backchannel():
    return (
        state == State.SHORT_PAUSE
        and not elevenlabs_output_active
        and len(current_buffer.split()) >= 3
        and time_since_last_backchannel > BACKCHANNEL_COOLDOWN
    )
```

### Backchannel Execution

```python
if should_fire_backchannel():
    response = select_from_weighted_pool(current_buffer)
    send_to_elevenlabs(response)
    reset_backchannel_cooldown()
```

---

## 6. ElevenLabs Output Layer

### Role

ElevenLabs converts text to speech. It receives streamed tokens from Claude and outputs audio chunks that publish to the agent's audio track.

### Gatekeeper Function

The output layer is the gatekeeper for all agent audio. Before streaming anything to ElevenLabs:

```python
def can_output():
    return not elevenlabs_output_active

def wait_for_output_idle():
    while elevenlabs_output_active:
        await asyncio.sleep(10)  # 10ms polling
```

### Output Priority

1. **Deepgram transcribing** = Hard interrupt. Stop all output immediately.
2. **ElevenLabs active** = Soft lock. New output waits.
3. **Backchannels and Claude responses** = Must wait for idle output layer.

### Position Tracking

Track output position for interruption handling:

```python
class ElevenLabsOutput:
    def __init__(self):
        self.active = False
        self.total_text = ""
        self.output_position = 0  # Characters actually spoken

    def on_audio_chunk_sent(self, chunk_text_length):
        self.output_position += chunk_text_length

    def get_spoken_text(self):
        return self.total_text[:self.output_position]

    def get_intended_text(self):
        return self.total_text
```

---

## 7. Universal Prompt Format

### Structure

Every LLM invocation uses a consistent format:

```xml
<agent_previous>
  <intended>{full text agent meant to say}</intended>
  <spoken>{text that was actually outputted as audio}</spoken>
</agent_previous>

<user_input>{current_buffer}</user_input>
```

### Example: No Interruption

```xml
<agent_previous>
  <intended>I can help you schedule that appointment for Tuesday at 3pm.</intended>
  <spoken>I can help you schedule that appointment for Tuesday at 3pm.</spoken>
</agent_previous>

<user_input>Actually, can we do Wednesday instead?</user_input>
```

### Example: User Interrupted Agent

```xml
<agent_previous>
  <intended>I can help you schedule that appointment for Tuesday at 3pm. Would you like me to send a confirmation to your email?</intended>
  <spoken>I can help you schedule that appointment for Tuesday at</spoken>
</agent_previous>

<user_input>Wait, I need to check my calendar first.</user_input>
```

### Example: User Interrupted During Prompting

When user speaks again before Claude responds:

```xml
<agent_previous>
  <intended></intended>
  <spoken></spoken>
</agent_previous>

<interrupted_context>{what user initially said before prompting}</interrupted_context>

<user_input>{what user added after interrupting}</user_input>
```

### Conversation History

Stack these blocks for full conversation context:

```xml
<turn index="1">
  <agent_previous>
    <intended>Hello, this is ThreadiaVA. How can I help you today?</intended>
    <spoken>Hello, this is ThreadiaVA. How can I help you today?</spoken>
  </agent_previous>
  <user_input>Hi, I need to schedule an appointment.</user_input>
</turn>

<turn index="2">
  <agent_previous>
    <intended>I'd be happy to help you schedule an appointment. What day works best for you?</intended>
    <spoken>I'd be happy to help you schedule an appointment. What day</spoken>
  </agent_previous>
  <user_input>Tuesday, wait no, let me check... Wednesday would be better.</user_input>
</turn>

<current_turn>
  <agent_previous>
    <intended></intended>
    <spoken></spoken>
  </agent_previous>
  <user_input>Yeah Wednesday at 3pm if possible.</user_input>
</current_turn>
```

---

## 8. Interruption Handling

### Detection

Interruption is detected when Deepgram emits words while in PROMPTING or AGENT_SPEAKING state.

```python
def on_deepgram_words(words):
    if state in [State.PROMPTING, State.AGENT_SPEAKING]:
        handle_interruption()

    # Always accumulate (interruption speech goes to buffer)
    current_buffer += words
```

### Interruption During PROMPTING

```python
def handle_interruption_prompting():
    # Cancel the Claude request
    cancel_claude_request()

    # interrupted_context already holds what we tried to send
    # (it was set when we entered PROMPTING)

    # Transition to listening
    state = State.LISTENING

    # Continue accumulating into current_buffer
```

### Interruption During AGENT_SPEAKING

```python
def handle_interruption_speaking():
    # Capture what was intended vs spoken
    intended = agent_response_buffer
    spoken = elevenlabs_output.get_spoken_text()

    # Store for next prompt
    agent_interrupted_context = {
        "intended": intended,
        "spoken": spoken
    }

    # Stop output
    stop_elevenlabs_output()

    # Transition to listening
    state = State.LISTENING

    # Continue accumulating into current_buffer
```

---

## 9. Timing and Configuration Constants

### Flux-Managed (No Manual Timers)

With Flux, turn detection is handled by Deepgram's ML model, not manual silence timers:

| Flux Parameter | Default | Purpose |
|----------------|---------|---------|
| `eot_threshold` | 0.7 | Confidence threshold for `EndOfTurn` event |
| `eager_eot_threshold` | 0.5 | Confidence threshold for `EagerEndOfTurn` event |
| `eot_timeout_ms` | 1000 | Force `EndOfTurn` after this silence (fallback) |

**Turn detection latency**: ~260ms (Flux internal processing)

### Application Constants (Still Required)

| Constant | Value | Purpose |
|----------|-------|---------|
| `BACKCHANNEL_COOLDOWN` | 3000ms | Minimum time between backchannels |
| `MIN_WORDS_FOR_BACKCHANNEL` | 3 | User must say at least this many words |
| `OUTPUT_IDLE_POLL_INTERVAL` | 10ms | How often to check if output layer is idle |
| `EAGER_LLM_TIMEOUT` | 5000ms | Max wait for speculative LLM before cancelling |

### Deprecated (Legacy Nova-3 Only)

These constants are only used when `model=nova-3` (manual turn detection mode):

| Constant | Value | Purpose |
|----------|-------|---------|
| `SHORT_PAUSE_THRESHOLD` | 200ms | Silence to enter SHORT_PAUSE |
| `END_OF_TURN_THRESHOLD` | 400ms | Silence to trigger prompting |

---

## 10. Component Integration Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                        LIVEKIT CLOUD                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   Twilio    │───▶│  SIP/RTP    │───▶│       Room          │  │
│  │  (PSTN)     │    │  Endpoint   │    │  ┌──────┐ ┌──────┐  │  │
│  └─────────────┘    └─────────────┘    │  │Caller│ │Agent │  │  │
│                                        │  └──┬───┘ └──┬───┘  │  │
│                                        └─────┼────────┼──────┘  │
└──────────────────────────────────────────────┼────────┼─────────┘
                                               │        │
                    ┌──────────────────────────┘        │
                    │ Audio Track                       │ Audio Track
                    ▼                                   ▲
┌─────────────────────────────────────────────────────────────────┐
│                         AGENT CODE                              │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  Deepgram   │───▶│   State     │───▶│   Claude (LLM)      │  │
│  │  (STT)      │    │  Machine    │    │                     │  │
│  └─────────────┘    └──────┬──────┘    └──────────┬──────────┘  │
│        │                   │                      │             │
│        │ Transcripts       │ Decisions            │ Tokens      │
│        │ Events            │                      │             │
│        ▼                   ▼                      ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  Buffers    │    │ Backchannel │    │   ElevenLabs        │  │
│  │             │    │   System    │───▶│   (TTS)             │  │
│  └─────────────┘    └─────────────┘    └──────────┬──────────┘  │
│                                                   │             │
│                                                   │ Audio       │
│                                                   ▼             │
│                                        ┌─────────────────────┐  │
│                                        │  Output to Caller   │  │
│                                        └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11. Implementation Checklist

### LiveKit Setup
- [ ] Create LiveKit Cloud account
- [ ] Configure SIP trunk for Twilio integration
- [ ] Set up dispatch rules for agent routing
- [ ] Deploy agent container to LiveKit Cloud

### Deepgram Flux Integration
- [x] Configure Deepgram plugin with `model=flux-general-en`
- [x] Use `/v2/listen` endpoint (not `/v1/listen`)
- [x] Set `eot_threshold` and `eager_eot_threshold` parameters
- [x] Implement `StartOfTurn` handler (barge-in detection)
- [x] Implement `EagerEndOfTurn` handler (backchannel + eager LLM)
- [x] Implement `TurnResumed` handler (cancel eager LLM)
- [x] Implement `EndOfTurn` handler (confirm/send LLM)
- [x] Implement `Update` handler (transcript accumulation)

### State Machine (Flux Event-Driven)
- [x] Implement 4-state enum: LISTENING, SHORT_PAUSE, PROMPTING, AGENT_SPEAKING
- [x] All transitions driven by Flux events (no manual timers)
- [x] Master guard blocks all events in AGENT_SPEAKING except barge-in
- [x] Handle eager LLM optimization (speculative calls)
- [x] Handle TurnResumed cancellation

### Buffer Management
- [x] Implement `current_buffer` accumulation
- [x] Implement `interrupted_context` preservation
- [x] Track `agent_response_buffer` during output
- [x] Track `agent_output_position` for interruption slicing
- [x] Reset turn state after each turn completes

### Backchannel System
- [x] Create weighted response pools by word count
- [x] Fire backchannel on `EagerEndOfTurn` event
- [x] Implement cooldown timer
- [x] Integrate with TTS output layer

### LLM Integration (Claude / AWS Bedrock)
- [x] Direct Anthropic API client with streaming
- [x] AWS Bedrock client with converse_stream API
- [x] Factory function for backend selection
- [x] Eager LLM optimization (start on EagerEndOfTurn)
- [x] Skip duplicate LLM on EndOfTurn if eager already sent
- [x] Cancel in-flight requests on interruption

### TTS Integration (ElevenLabs)
- [ ] Configure TTS streaming
- [ ] Implement output active tracking
- [ ] Add position tracking for interruptions
- [ ] Implement gatekeeper pattern

### Prompt Engineering
- [x] Universal prompt format with intended/spoken
- [x] Conversation history stacking
- [x] Interrupted context handling
- [x] Transfer function tool definition

### Interruption Handling
- [x] Detect barge-in via `StartOfTurn` during AGENT_SPEAKING
- [x] Detect interruption during PROMPTING
- [x] Cancel LLM request on interruption
- [x] Stop TTS output on barge-in
- [x] Preserve intended/spoken context

---

## 12. Error Handling Considerations

### Deepgram Latency Spike
If Deepgram hasn't emitted words for an unusually long time despite audio flowing:
- Track last Deepgram emission timestamp
- If END_OF_TURN would fire but buffer is empty/stale, wait
- Don't prompt Claude on empty context

### ElevenLabs Latency Spike
If Claude tokens are streaming but ElevenLabs hasn't started output:
- Track time from first token to first audio
- If threshold exceeded (800ms), consider injecting filler: "Let me think..."
- Signals presence, buys time

### Network Failures
- Implement reconnection logic for Deepgram WebSocket
- Implement reconnection logic for ElevenLabs WebSocket
- Handle Claude API timeouts gracefully
- Log all failures for debugging

---

## 13. Future Enhancements

### v2 Considerations
- Audio event classification (coughs, laughter) for contextual responses
- Sentiment analysis on transcripts for tone adjustment
- Dynamic timing thresholds based on conversation pace
- Multi-language support with language detection

---

## 14. Configuration Architecture

### Static vs Dynamic Configuration

Configuration is split into two types:

| Type | Source | Contents |
|------|--------|----------|
| **Static** | `.env` file | API keys, infrastructure credentials, timing defaults |
| **Dynamic** | Frontend payload | Voice options, agent name, system prompt, lead context |

### Static Configuration (.env)

```env
# API Keys (required)
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=xxx
LIVEKIT_API_SECRET=xxx
DEEPGRAM_API_KEY=xxx
ELEVENLABS_API_KEY=xxx
ANTHROPIC_API_KEY=xxx

# Defaults (can be tuned)
DEEPGRAM_MODEL=nova-2
ANTHROPIC_MODEL=claude-3-5-haiku-20241022
SHORT_PAUSE_THRESHOLD_MS=200
END_OF_TURN_THRESHOLD_MS=400
```

### Dynamic Configuration (Per-Call Payload)

Comes from frontend via SIP headers or room metadata:

```json
{
  "session_id": "sess_xxx",
  "lead_id": "lead_xxx",
  "business_id": "biz_xxx",
  "voice_id": "elevenlabs_voice_id_selected_by_business",
  "voice_stability": 0.5,
  "voice_similarity_boost": 0.75,
  "agent_name": "Sarah",
  "system_prompt": "You are a sales representative...",
  "greeting_message": "Hello {lead_name}, this is {agent_name}...",
  "lead_name": "John Doe",
  "lead_pertinent_info": "Interested in enterprise plan...",
  "business_name": "Acme Corp",
  "business_transfer_number": "+1234567890",
  "backchannels_enabled": true,
  "interruption_handling": true
}
```

### Why This Split?

- **Static config**: Same across all calls, contains secrets, loaded once at startup
- **Dynamic config**: Different per call, no secrets, comes from DynamoDB/frontend

---

## 15. Edge Cases and State Machine Behavior

This section documents how the Flux-driven state machine handles edge cases.

### Edge Case 1: Rapid-Fire User Interruptions

**Scenario**: User speaks → EagerEndOfTurn → user immediately speaks again → EagerEndOfTurn → speaks again

**Behavior**:
1. First `EagerEndOfTurn` → SHORT_PAUSE, starts eager LLM
2. Second `StartOfTurn` → TurnResumed cancels eager LLM, back to LISTENING
3. Second `EagerEndOfTurn` → SHORT_PAUSE, new eager LLM
4. Process repeats until user actually stops

**Why it works**: Flux guarantees `TurnResumed` always follows `EagerEndOfTurn` if user continues. The eager LLM task is cancelled cleanly.

### Edge Case 2: EagerEndOfTurn Without TurnResumed (User Done)

**Scenario**: User speaks → EagerEndOfTurn → silence → EndOfTurn

**Behavior**:
1. `EagerEndOfTurn` → SHORT_PAUSE, starts eager LLM, fires backchannel
2. Eager LLM completes and streams tokens → AGENT_SPEAKING, sets `_eager_llm_sent = True`
3. `EndOfTurn` arrives → Check `_eager_llm_sent`, skip duplicate LLM call
4. Agent continues speaking from eager response

**Why it works**: Flux guarantees `EndOfTurn` transcript matches `EagerEndOfTurn` transcript. No need to re-prompt.

### Edge Case 3: EndOfTurn Before Eager LLM Completes

**Scenario**: User speaks → EagerEndOfTurn → EndOfTurn fires quickly → eager LLM still pending

**Behavior**:
1. `EagerEndOfTurn` → SHORT_PAUSE, starts eager LLM (pending)
2. `EndOfTurn` → PROMPTING, cancels pending eager LLM
3. `_eager_llm_sent` is False → send final LLM call normally

**Why it works**: The eager task is cancelled via `_cancel_eager_llm()` before checking the flag.

### Edge Case 4: User Barge-In During Agent Speaking

**Scenario**: Agent is speaking → user starts talking → `StartOfTurn` event

**Behavior**:
1. AGENT_SPEAKING state, Master Guard active
2. `StartOfTurn` detected → only allowed event in this state
3. Stop TTS immediately, capture `intended` vs `spoken`
4. Transition to LISTENING
5. Accumulate new user speech
6. Next prompt includes interrupted context

**Why it works**: Master Guard blocks all events except `StartOfTurn` (barge-in) and `output_complete`.

### Edge Case 5: Agent Speaking Completes During User Speech

**Scenario**: Agent finishes speaking while user is already talking (overlap)

**Behavior**:
1. AGENT_SPEAKING → `output_complete` → LISTENING
2. But user speech already triggered `StartOfTurn` before completion
3. State is LISTENING, user speech accumulates normally
4. `EagerEndOfTurn` → normal turn detection flow

**Why it works**: The `output_complete` event only fires when TTS truly finishes. If user already triggered barge-in, we're in LISTENING and `output_complete` is ignored.

### Edge Case 6: Empty Buffer on EndOfTurn

**Scenario**: Somehow `EndOfTurn` fires but transcript buffer is empty

**Behavior**:
1. `EndOfTurn` → PROMPTING
2. Buffer check: `if not self._pending_transcript.strip():`
3. Log warning, skip LLM call
4. Return to LISTENING

**Why it works**: Flux guarantee states `EagerEndOfTurn` always has nonempty transcript, so `EndOfTurn` should too. But defensive check prevents empty prompts.

### Edge Case 7: LLM Times Out During PROMPTING

**Scenario**: Claude/Bedrock takes too long to respond

**Behavior**:
1. PROMPTING state, LLM call in progress
2. Timeout triggers → cancel request
3. Log error with context
4. Inject filler: "I'm having trouble connecting. One moment please."
5. Retry once, then gracefully degrade

**Why it works**: LLM client has built-in timeout. State machine stays consistent.

### Edge Case 8: Multiple TurnResumed Events

**Scenario**: Flux sends multiple `TurnResumed` events (shouldn't happen per spec)

**Behavior**:
1. First `TurnResumed` → cancels eager LLM, back to LISTENING
2. Second `TurnResumed` → state is already LISTENING
3. No-op, already in correct state

**Why it works**: State machine is idempotent for same-state transitions.

### Edge Case 9: Backchannel During Eager LLM Streaming

**Scenario**: Backchannel fires, then eager LLM tokens start arriving

**Behavior**:
1. `EagerEndOfTurn` → SHORT_PAUSE
2. Backchannel fires asynchronously (short audio)
3. Eager LLM tokens arrive
4. Check: is TTS still outputting backchannel?
5. Wait for backchannel to finish → then stream LLM response
6. Transition to AGENT_SPEAKING

**Why it works**: TTS gatekeeper ensures no overlapping audio. Backchannel completes quickly (300-500ms).

### Edge Case 10: Transfer Function During Conversation

**Scenario**: LLM decides to transfer, calls `transfer_to_human` tool

**Behavior**:
1. LLM streams response including tool_use block
2. Tool call detected: `transfer_to_human`
3. Extract arguments: `qualification_score`, `transfer_reason`
4. Fire `on_transfer_request` callback
5. SIP REFER to business transfer number
6. Session ends, recording saved

**Why it works**: Tool calls are detected during token streaming. Transfer is a terminal action.

### State Machine Invariants

These invariants must always hold:

1. **Single Active State**: Exactly one state at any time
2. **Master Guard**: AGENT_SPEAKING only exits via `StartOfTurn` or `output_complete`
3. **No Orphan Tasks**: Eager LLM cancelled on any transition out of SHORT_PAUSE
4. **Buffer Preservation**: `interrupted_context` survives until successful LLM response
5. **Event Ordering**: Flux events processed in order (WebSocket guarantees)

---

## Changelog

### Version 1.2 (December 4, 2024)
- **Major Update**: Migrated from manual timing to Deepgram Flux event-driven architecture
- **Updated**: Section 2 (Signal Layer) - Complete Flux documentation with events, guarantees, and configuration
- **Updated**: Section 3 (State Machine) - All transitions now driven by Flux events, not manual timers
- **Updated**: Section 9 (Timing Constants) - Flux-managed vs application constants
- **Updated**: Section 11 (Implementation Checklist) - Reflects Flux-based implementation status
- **Added**: Section 15 (Edge Cases) - 10 edge case scenarios with state machine behavior
- **Added**: Eager LLM optimization documentation (speculative calls on EagerEndOfTurn)
- **Added**: AWS Bedrock support as alternative LLM backend
- **Fixed**: Eager LLM race condition (skip duplicate on EndOfTurn if eager already sent)

### Version 1.1 (December 3, 2024)
- **Added**: Configuration Architecture section (Section 14)
- **Clarified**: Voice options come from frontend payload, not .env file
- **Added**: CallConfig dataclass for per-call dynamic settings
- **Added**: Support for SIP headers and room metadata parsing

### Version 1.0 (December 2024)
- Initial specification document

---

*Document Version: 1.2*
*Last Updated: December 4, 2025*
*Author: ThreadiaVA Engineering*
