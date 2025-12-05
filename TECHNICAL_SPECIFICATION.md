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

### Buffers (Word-Level Tracking)

| Buffer | Purpose |
|--------|---------|
| `current_buffer` | Accumulates user speech in current turn (word list) |
| `interrupted_context` | Stores prompt that was interrupted before Claude responded |
| `agent_response_words` | Word list from LLM output (intended text) |
| `agent_spoken_words` | Word list confirmed spoken by TTS (spoken text) |

### Word-Level TTS Tracking

Two parallel word lists track LLM output vs TTS playback:

```
LLM streams tokens  →  agent_response_words: ["Hello", "I", "can", "help", "you", "with", "..."]
TTS plays audio     →  agent_spoken_words:   ["Hello", "I", "can", "help"]  # played so far
User interrupts     →  intended: all words, spoken: spoken words only
```

### Buffer Flow: Normal Turn

```
User speaks → current_buffer accumulates words
End of turn → current_buffer sent to Claude
Claude responds → agent_response_words fills (word by word)
TTS plays → agent_spoken_words tracks what was output
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
Agent speaking → agent_response_words = full intended response
              → agent_spoken_words = words TTS has played
User interrupts → intended = agent_response_words (all)
               → spoken = agent_spoken_words (only played)
               → store both for context
               → accumulate new speech to current_buffer
End of turn → send full context (intended, spoken, user input) to Claude
```

### TTS Integration API

```python
# LLM streams tokens
buffers.append_agent_token(token)  # Auto-splits into words

# TTS reports words played
buffers.mark_words_spoken(["Hello", "I", "can"])  # By word list
# OR
buffers.mark_words_spoken_by_count(4)  # By count

# On interruption
context = buffers.handle_speaking_interruption()
# Returns: AgentInterruptedContext(
#   intended="Hello I can help you with...",
#   spoken="Hello I can",
#   intended_words=["Hello", "I", "can", "help", "you", "with", "..."],
#   spoken_words=["Hello", "I", "can"]
# )
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

## 6. ElevenLabs Output Layer (Multi-Context WebSocket)

### Role

ElevenLabs converts text to speech. It receives streamed tokens from Claude via **Multi-Context WebSocket** and outputs audio chunks that publish to the agent's audio track.

### Why Multi-Context WebSocket (Not HTTP Streaming)

| Approach | Problem | Solution |
|----------|---------|----------|
| HTTP Streaming | Requires complete text upfront | Multi-Context allows token streaming |
| Token-by-token HTTP | Choppy, unnatural prosody | context_id groups tokens for prosodic consistency |
| Single WebSocket | Can't handle barge-in cleanly | Multi-context allows closing interrupted contexts |

### Architecture

```
                    Single WebSocket Connection (per session)
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
      context_1               context_2               context_3
   (response #1)           (response #2)           (backchannel)
            │                       │                       │
    "Hello, I can"          "Sure, let me"              "Mhmm"
    "help you with"          (interrupted)
    "scheduling..."
```

### WebSocket Endpoint

```
wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/multi-stream-input?model_id={model_id}
```

**Models**: `eleven_turbo_v2_5` (fast), `eleven_flash_v2_5` (balanced)

**Note**: Multi-context is NOT available for `eleven_v3` model.

### Context Lifecycle

Each LLM response uses one context_id for prosodic consistency:

```python
# 1. Start new context for response
context_id = await tts.start_response()

# 2. Stream tokens as they arrive from LLM
for token in llm_stream:
    await tts.stream_token(token)  # voice_settings on first message only

# 3. End context when response complete
await tts.end_response()
```

### Voice Settings (First Message Only)

Per ElevenLabs docs, `voice_settings` only sent on first message per context:

```python
# First message in context includes voice_settings
message = {
    "text": token,
    "context_id": context_id,
    "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.75
    }
}

# Subsequent messages omit voice_settings
message = {
    "text": token,
    "context_id": context_id
}
```

### Sentence-Level Flushing

Flush at sentence boundaries for natural speech:

```python
# After streaming tokens
if token.endswith(('.', '!', '?')):
    await ws.send_json({
        "context_id": context_id,
        "flush": True
    })
```

**Why flush?** ElevenLabs buffers text for context. Flushing forces generation, improving responsiveness while maintaining quality.

### Handling Barge-In (Interruption)

When user speaks during AGENT_SPEAKING:

```python
async def handle_barge_in():
    # Close current context immediately (non-blocking)
    await ws.send_json({
        "context_id": current_context_id,
        "close_context": True
    })

    # Capture intended vs spoken for context
    intended_words = words_sent.copy()
    spoken_words = words_spoken.copy()

    # Reset state
    active = False
```

### Keep-Alive for Long Pauses

Contexts timeout after 20 seconds of inactivity. During LLM thinking:

```python
async def keep_context_alive():
    await ws.send_json({
        "context_id": context_id,
        "text": ""  # Empty text resets timeout clock
    })
```

### Word-Level Tracking

Track words sent to TTS vs words actually spoken for accurate interruption context:

```python
# As tokens stream
words_sent.extend(token.split())

# As audio chunks received
words_spoken.append(pending_words.pop(0))

# On interruption
intended = " ".join(words_sent)      # Full response
spoken = " ".join(words_spoken)      # What caller heard
```

### Response Handling

ElevenLabs uses **camelCase** in responses:

```python
data = json.loads(message)
context_id = data.get("contextId")  # Note: camelCase

if data.get("audio"):
    # Binary audio data (base64 in JSON responses)
    pass

if data.get("is_final"):
    # Context audio generation complete
    # Mark all pending words as spoken
    pass
```

### Gatekeeper Function

The output layer is the gatekeeper for all agent audio:

```python
def can_output():
    return not state.active

async def wait_for_idle(timeout_ms=10000):
    while state.active and elapsed < timeout_ms:
        await asyncio.sleep(0.01)  # 10ms polling
```

### Output Priority

1. **Deepgram StartOfTurn** = Hard interrupt. Close context immediately.
2. **ElevenLabs active** = Soft lock. New output waits.
3. **Backchannels** = Use HTTP streaming (faster for short text).
4. **Claude responses** = Use Multi-Context WebSocket.

### Backchannel via HTTP

Backchannels use HTTP streaming for lower latency (no context overhead):

```python
async def speak_backchannel(text):
    if state.active:
        return False  # Skip if main response playing

    # HTTP endpoint for short utterances
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    async with session.post(url, json={"text": text, ...}) as response:
        async for chunk in response.content.iter_chunked(4096):
            publish_audio(chunk)
```

### Connection Lifecycle

```python
# On session start
await tts.connect()  # Single WebSocket for entire call

# During call
for response in responses:
    await tts.start_response()
    for token in tokens:
        await tts.stream_token(token)
    await tts.end_response()

# On session end
await tts.disconnect()  # Sends close_socket: true
```

### Limits and Timeouts

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max concurrent contexts | 5 | Per WebSocket connection |
| Inactivity timeout | 20s | Default, up to 180s configurable |
| Max message size | 16MB | For audio responses |

---

## 7. Emotion Detection Layer (Hume AI + SharedAudioBuffer)

### Overview

The Emotion Detection Layer provides real-time emotional context from user speech prosody. It uses Hume AI's Expression Measurement API to analyze voice characteristics (pitch, tempo, intensity) and correlate emotions with Deepgram transcripts using a shared `context_id`.

### Why Emotion Detection?

| Without Emotion | With Emotion |
|-----------------|--------------|
| "I need help with my order" | `<frustration confidence="0.78">I need help with my order</frustration>` |
| LLM responds neutrally | LLM responds empathetically, acknowledges frustration |
| Generic tone | Tailored emotional response |

### Architecture: Dual-Stream Processing

```
                         User Audio Stream
                                │
                    ┌───────────┴───────────┐
                    │   SharedAudioBuffer   │
                    │   (generates context_id)
                    └───────────┬───────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  Deepgram   │     │  Hume AI    │     │   Audio     │
    │  (STT)      │     │  (Prosody)  │     │   Context   │
    │             │     │             │     │   Storage   │
    └──────┬──────┘     └──────┬──────┘     └─────────────┘
           │                   │
           │ transcript        │ emotions
           │                   │
           ▼                   ▼
    ┌──────────────────────────────────────┐
    │    EndOfTurn: Correlate by context_id │
    │    Output: <emotion>transcript</emotion> │
    └──────────────────────────────────────┘
```

### SharedAudioBuffer: The Central Coordinator

The `SharedAudioBuffer` is the central audio routing layer that:

1. **Generates `context_id`** on first audio chunk of a turn
2. **Routes audio** to both Deepgram (real-time, every chunk) and Hume (batched, 500ms intervals)
3. **Correlates results** by `context_id`
4. **At EndOfTurn**, returns emotion-tagged transcript

```python
class SharedAudioBuffer:
    def receive_audio(self, chunk: bytes) -> str:
        """
        Receive audio from LiveKit.
        Returns context_id for this turn.
        """
        # Create context on first chunk
        if not self._current_context:
            context_id = self._generate_context_id()
            self._current_context = AudioContext(context_id=context_id)

        # Route to Deepgram (every chunk)
        await self._on_audio_for_deepgram(context_id, chunk)

        # Batch for Hume (500ms intervals)
        self._hume_buffer.append(chunk)
        self._maybe_send_to_hume(context_id)

        return context_id

    def end_turn(self, context_id: str, final_transcript: str) -> str:
        """
        End turn and return emotion-tagged transcript.
        ZERO LATENCY: Reads latest_emotion directly from HumeHandler.
        """
        # Read emotion (already updated via continuous prediction)
        if self._hume_handler and self._hume_handler.has_emotion:
            emotion, confidence = self._hume_handler.get_top_emotion()
            category = map_to_category(emotion)
            return f'<{category} confidence="{confidence:.2f}">{final_transcript}</{category}>'

        # Graceful degradation: return neutral if Hume unavailable
        return f'<neutral confidence="0.50">{final_transcript}</neutral>'
```

### HumeHandler: Real-Time Prosody Analysis

Connects to Hume AI's Expression Measurement WebSocket API for continuous emotion detection:

```python
class HumeHandler:
    HUME_WS_URL = "wss://api.hume.ai/v0/stream/models"

    # Continuous prediction: always-updated latest emotion
    _latest_emotion: Optional[Dict[str, float]] = None

    async def send_audio(self, audio_bytes: bytes) -> bool:
        """Send audio for prosody analysis."""
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

        message = {
            "data": audio_b64,
            "models": {"prosody": {}},
            "raw_text": False,
            "stream_window_ms": 500
        }

        await self._ws.send_json(message)

    def get_top_emotion(self) -> Tuple[str, float]:
        """Get top emotion from latest prediction."""
        if not self._latest_emotion:
            return ("neutral", 0.5)
        return max(self._latest_emotion.items(), key=lambda x: x[1])
```

### Continuous Prediction Pattern (Zero Latency)

Unlike request-response patterns that add latency, we use **continuous prediction**:

```
                    Audio streaming...
                         │
    ┌──────────────────────────────────────────────┐
    │                  Timeline                     │
    ├──────┬──────┬──────┬──────┬──────┬──────┬────┤
    │  0ms │ 500ms│1000ms│1500ms│2000ms│ EOT  │    │
    │      │      │      │      │      │      │    │
    │ Hume │ Hume │ Hume │ Hume │ Hume │ Read │    │
    │ send │ recv │ send │ recv │ send │ latest    │
    │      │ updt │      │ updt │      │ emotion   │
    └──────┴──────┴──────┴──────┴──────┴──────┴────┘
                                         ▲
                                         │
                                    Zero wait!
```

At EndOfTurn, we simply read `_latest_emotion` - no waiting required.

### Emotion Categories (Hume's 48 → Simplified 10)

Hume detects 48 prosody emotions. We map to simplified categories for LLM prompting:

| Category | Hume Emotions |
|----------|---------------|
| `joy` | Amusement, Joy, Excitement, Interest, Satisfaction |
| `calm` | Calmness, Contentment, Relief, Serenity |
| `interest` | Interest, Curiosity, Concentration, Contemplation |
| `frustration` | Frustration, Annoyance, Irritation |
| `anger` | Anger, Contempt, Disgust |
| `sadness` | Sadness, Disappointment, Distress |
| `anxiety` | Anxiety, Fear, Nervousness, Worry |
| `confusion` | Confusion, Doubt, Awkwardness |
| `surprise` | Surprise, Awe, Amazement |
| `neutral` | Neutral, Boredom, Tiredness |

### Integration with State Machine

Emotion detection integrates seamlessly with Flux turn events:

```python
# On StartOfTurn
self._shared_buffer.cancel_turn()  # Clear previous context
self._hume_handler.clear_emotion()  # Reset emotion state

# On audio chunk received
context_id = self._shared_buffer.receive_audio(chunk)

# On EndOfTurn
tagged_transcript = self._shared_buffer.end_turn(
    context_id=context_id,
    final_transcript=deepgram_transcript
)
# Output: "<calm confidence=\"0.72\">I need to schedule an appointment</calm>"

# Send to LLM with emotion context
await self._llm_client.generate_response(tagged_transcript)
```

### Hume API Configuration

```python
@dataclass
class HumeConfig:
    api_key: str
    models: List[str] = ["prosody"]  # Speech emotion analysis
```

Connection URL: `wss://api.hume.ai/v0/stream/models`

Authentication: `X-Hume-Api-Key` header

### Graceful Degradation

If Hume is unavailable, the system degrades gracefully:

1. **Circuit breaker opens** after 5 consecutive failures
2. **Degraded mode active**: Skip sending audio to Hume
3. **Neutral fallback**: Return `<neutral confidence="0.50">transcript</neutral>`
4. **Auto-recovery**: Circuit breaker allows retry after 30 seconds

```python
# In SharedAudioBuffer.end_turn()
if self._hume_handler.is_degraded:
    # Graceful degradation: return neutral
    return f'<neutral confidence="0.50">{final_transcript}</neutral>'
```

---

## 8. Production Resilience Layer

### Overview

The Production Resilience Layer provides fault tolerance for external service integrations (Hume AI, ElevenLabs). It ensures the voice pipeline continues operating even when dependencies fail.

### Resilience Patterns

| Pattern | Purpose | Implementation |
|---------|---------|----------------|
| Circuit Breaker | Fail fast after repeated failures | 5 failures → open, 30s timeout → half-open |
| Exponential Backoff | Prevent thundering herd | 500ms → 5s with jitter |
| Graceful Degradation | Continue without failed service | Return neutral emotion / skip TTS |
| Health Checks | Monitor and recover | Periodic health_check() with auto-recovery |

### Circuit Breaker

```
         Failures < threshold                    timeout elapsed
              ┌────────┐                           ┌────────┐
              │        │                           │        │
              ▼        │                           ▼        │
         ┌────────┐    │    failure threshold  ┌────────────┐
 ───────▶│ CLOSED │────┴─────────────────────▶│   OPEN     │
         └────┬───┘                            └─────┬──────┘
              │                                      │
              │ success                              │ timeout
              │                                      │ elapsed
              │         ┌───────────────┐            │
              └─────────│  HALF-OPEN    │◀───────────┘
                        └───────┬───────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
               success                  failure
                    │                       │
                    ▼                       ▼
               ┌────────┐             ┌────────┐
               │ CLOSED │             │  OPEN  │
               └────────┘             └────────┘
```

### Configuration

```python
@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5      # Failures before opening
    success_threshold: int = 2      # Successes to close from half-open
    timeout_seconds: float = 30.0   # Time before attempting recovery

@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay_ms: int = 500
    max_delay_ms: int = 5000
    exponential_base: float = 2.0
    jitter_factor: float = 0.1      # 10% randomization
```

### Error Classification

Errors are classified to determine retry strategy:

| Error Type | Action | Examples |
|------------|--------|----------|
| `TRANSIENT` | Retry immediately | Network timeout, connection reset |
| `THROTTLED` | Back off significantly | Rate limit (429), too many requests |
| `PERMANENT` | Don't retry | Auth failed (401/403), bad request (400) |
| `UNKNOWN` | Retry with caution | Unexpected errors |

```python
def classify_error(error: Exception) -> ClassifiedError:
    """Classify error to determine retry strategy."""
    if is_rate_limit(error):
        return ClassifiedError(ErrorType.THROTTLED, should_retry=True)
    if is_auth_error(error):
        return ClassifiedError(ErrorType.PERMANENT, should_retry=False)
    if is_connection_error(error):
        return ClassifiedError(ErrorType.TRANSIENT, should_retry=True)
    return ClassifiedError(ErrorType.UNKNOWN, should_retry=True)
```

### Resilience in HumeHandler

```python
class HumeHandler:
    CONNECT_TIMEOUT_SECONDS = 10.0
    SEND_TIMEOUT_SECONDS = 5.0

    def __init__(self, enable_circuit_breaker: bool = True):
        if enable_circuit_breaker:
            self._circuit_breaker = CircuitBreaker(
                name="hume",
                config=CircuitBreakerConfig(
                    failure_threshold=5,
                    success_threshold=2,
                    timeout_seconds=30.0
                )
            )

        self._retry_config = RetryConfig(
            max_retries=3,
            base_delay_ms=500,
            max_delay_ms=5000
        )

    async def connect(self) -> bool:
        # Check circuit breaker first
        if not self._circuit_breaker.can_execute():
            self._degraded = True
            return False

        # Retry with backoff
        await retry_with_backoff(
            self._do_connect,
            config=self._retry_config
        )

    async def send_audio(self, audio_bytes: bytes) -> bool:
        # Circuit breaker check
        if not self._circuit_breaker.can_execute():
            return False  # Graceful degradation

        # Send with timeout
        await asyncio.wait_for(
            self._ws.send_json(message),
            timeout=self.SEND_TIMEOUT_SECONDS
        )
```

### Resilience in TTSOutput

```python
class TTSOutput:
    CONNECT_TIMEOUT_SECONDS = 10.0
    SEND_TIMEOUT_SECONDS = 5.0
    HTTP_TIMEOUT_SECONDS = 15.0

    async def connect(self) -> bool:
        # Circuit breaker + retry with backoff
        if not self._circuit_breaker.can_execute():
            self._degraded = True
            return False

        await retry_with_backoff(
            self._do_connect,
            config=self._retry_config
        )

    async def stream_token(self, token: str) -> None:
        # Circuit breaker check
        if not self._circuit_breaker.can_execute():
            return  # Silent failure, don't block

        # Send with timeout
        await asyncio.wait_for(
            self._ws.send_json(message),
            timeout=self.SEND_TIMEOUT_SECONDS
        )
```

### Health Checks

Both handlers expose health check methods for monitoring:

```python
async def health_check(self) -> Dict[str, Any]:
    """Perform health check with auto-recovery."""
    health = {
        "service": "hume",
        "connected": self._connected,
        "degraded": self._degraded,
        "circuit_state": self.circuit_state,
        "consecutive_failures": self._consecutive_failures
    }

    # Attempt recovery if degraded but circuit closed
    if self._degraded and self._circuit_breaker.state == CircuitState.CLOSED:
        connected = await self.connect()
        if connected:
            self._degraded = False
            health["recovered"] = True

    return health
```

### Timeout Constants

| Operation | HumeHandler | TTSOutput |
|-----------|-------------|-----------|
| Connect | 10s | 10s |
| Send | 5s | 5s |
| HTTP Request | - | 15s |

---

## 9. Universal Prompt Format

### Structure

Every LLM invocation uses a consistent format with **emotion-tagged user input**:

```xml
<agent_previous>
  <intended>{full text agent meant to say}</intended>
  <spoken>{text that was actually outputted as audio}</spoken>
</agent_previous>

<user_input>
  <{emotion} confidence="{0.XX}">{current_buffer}</{emotion}>
</user_input>
```

### Emotion Tag Format

User input is wrapped with the detected emotion from Hume AI:

```xml
<{emotion_category} confidence="{confidence_score}">{transcript}</{emotion_category}>
```

**Categories**: `joy`, `calm`, `interest`, `frustration`, `anger`, `sadness`, `anxiety`, `confusion`, `surprise`, `neutral`

**Confidence**: 0.00 - 1.00 (higher = more confident)

### Example: Calm User (No Interruption)

```xml
<agent_previous>
  <intended>I can help you schedule that appointment for Tuesday at 3pm.</intended>
  <spoken>I can help you schedule that appointment for Tuesday at 3pm.</spoken>
</agent_previous>

<user_input>
  <calm confidence="0.72">Actually, can we do Wednesday instead?</calm>
</user_input>
```

### Example: Frustrated User Interrupted Agent

```xml
<agent_previous>
  <intended>I can help you schedule that appointment for Tuesday at 3pm. Would you like me to send a confirmation to your email?</intended>
  <spoken>I can help you schedule that appointment for Tuesday at</spoken>
</agent_previous>

<user_input>
  <frustration confidence="0.78">Wait, I need to check my calendar first.</frustration>
</user_input>
```

**Note**: The LLM can now detect the user's frustration and respond with appropriate empathy.

### Example: User Interrupted During Prompting

When user speaks again before Claude responds:

```xml
<agent_previous>
  <intended></intended>
  <spoken></spoken>
</agent_previous>

<interrupted_context>
  <calm confidence="0.65">{what user initially said before prompting}</calm>
</interrupted_context>

<user_input>
  <anxiety confidence="0.58">{what user added after interrupting}</anxiety>
</user_input>
```

### Example: Graceful Degradation (Hume Unavailable)

When Hume AI is unavailable, neutral emotion with 0.50 confidence is returned:

```xml
<user_input>
  <neutral confidence="0.50">I need help with my order.</neutral>
</user_input>
```

### Conversation History with Emotions

Stack these blocks for full conversation context:

```xml
<turn index="1">
  <agent_previous>
    <intended>Hello, this is ThreadiaVA. How can I help you today?</intended>
    <spoken>Hello, this is ThreadiaVA. How can I help you today?</spoken>
  </agent_previous>
  <user_input>
    <calm confidence="0.68">Hi, I need to schedule an appointment.</calm>
  </user_input>
</turn>

<turn index="2">
  <agent_previous>
    <intended>I'd be happy to help you schedule an appointment. What day works best for you?</intended>
    <spoken>I'd be happy to help you schedule an appointment. What day</spoken>
  </agent_previous>
  <user_input>
    <confusion confidence="0.71">Tuesday, wait no, let me check... Wednesday would be better.</confusion>
  </user_input>
</turn>

<current_turn>
  <agent_previous>
    <intended></intended>
    <spoken></spoken>
  </agent_previous>
  <user_input>
    <interest confidence="0.62">Yeah Wednesday at 3pm if possible.</interest>
  </user_input>
</current_turn>
```

### LLM System Prompt for Emotion Awareness

Include instructions for the LLM to leverage emotion tags:

```
You are a voice agent for {business_name}. The user's speech includes emotion tags
detected from their voice prosody. Use these to tailor your responses:

- <frustration>: Acknowledge their frustration, be more concise and solution-focused
- <anxiety>: Be reassuring, speak calmly, offer clear next steps
- <confusion>: Clarify and simplify, ask confirming questions
- <joy>: Match their energy, be enthusiastic
- <calm>: Maintain professional, conversational tone

The confidence score (0.0-1.0) indicates how certain the emotion detection is.
High confidence (>0.7) suggests strong emotional signal.
```

---

## 10. Interruption Handling

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

## 11. Timing and Configuration Constants

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

## 12. Component Integration Summary

### ThreadiaVA Engine Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LIVEKIT CLOUD                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────────┐  │
│  │   Twilio    │───▶│  SIP/RTP    │───▶│             Room                │  │
│  │  (PSTN)     │    │  Endpoint   │    │  ┌──────────┐  ┌──────────────┐ │  │
│  └─────────────┘    └─────────────┘    │  │  Caller  │  │    Agent     │ │  │
│                                        │  └────┬─────┘  └──────┬───────┘ │  │
│                                        └───────┼───────────────┼─────────┘  │
└────────────────────────────────────────────────┼───────────────┼────────────┘
                                                 │               │
                          ┌──────────────────────┘               │
                          │ Audio Track                          │ Audio Track
                          ▼                                      ▲
┌─────────────────────────────────────────────────────────────────────────────┐
│                         THREADIAVA ENGINE (Agent Code)                       │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      SharedAudioBuffer                                  │ │
│  │                   (generates context_id per turn)                       │ │
│  └───────────┬───────────────────────────────────────────────┬─────────────┘ │
│              │                                               │               │
│              │ Audio chunks                                  │ Audio batches │
│              ▼                                               ▼               │
│  ┌─────────────────────────┐                    ┌─────────────────────────┐  │
│  │      Deepgram Flux      │                    │       Hume AI           │  │
│  │         (STT)           │                    │      (Prosody)          │  │
│  │ StartOfTurn, EndOfTurn  │                    │   emotion detection     │  │
│  └───────────┬─────────────┘                    └───────────┬─────────────┘  │
│              │ Transcripts + Events                         │ Emotions       │
│              │                                              │                │
│              ▼                                              ▼                │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                           State Machine                                 │ │
│  │    LISTENING → SHORT_PAUSE → PROMPTING → AGENT_SPEAKING → LISTENING    │ │
│  └────────────────────────────────┬────────────────────────────────────────┘ │
│                                   │                                          │
│              ┌────────────────────┼────────────────────┐                     │
│              │                    │                    │                     │
│              ▼                    ▼                    ▼                     │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐            │
│  │   BufferManager │   │  Backchannel    │   │   LLM Client    │            │
│  │   (word-level)  │   │     System      │   │  (Claude/AWS)   │            │
│  └────────┬────────┘   └────────┬────────┘   └────────┬────────┘            │
│           │                     │                     │                      │
│           │ context             │ short utterances    │ tokens (stream)      │
│           ▼                     ▼                     ▼                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    ElevenLabs TTS (Multi-Context)                       │ │
│  │              circuit breaker + retry + graceful degradation             │ │
│  └───────────────────────────────────┬─────────────────────────────────────┘ │
│                                      │ Audio                                 │
│                                      ▼                                       │
│                            ┌─────────────────────┐                           │
│                            │  Publish to Caller  │                           │
│                            └─────────────────────┘                           │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                     Production Resilience Layer                         │ │
│  │   Circuit Breaker │ Exponential Backoff │ Graceful Degradation │ Health │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

```
Audio In → SharedAudioBuffer → context_id generated
                │
    ┌───────────┴───────────┐
    │                       │
    ▼                       ▼
Deepgram (STT)         Hume AI (Emotion)
    │                       │
    │ transcript            │ emotions dict
    │                       │
    ▼                       ▼
EndOfTurn ────────────────────────────▶ SharedAudioBuffer.end_turn()
                                              │
                                              ▼
                              <emotion>transcript</emotion>
                                              │
                                              ▼
                                        LLM (Claude)
                                              │
                                              ▼
                                        ElevenLabs TTS
                                              │
                                              ▼
                                        Audio Out
```

---

## 13. Implementation Checklist

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

### TTS Integration (ElevenLabs Multi-Context WebSocket)
- [x] Multi-Context WebSocket connection with `model_id` parameter
- [x] Context lifecycle management (`start_response`, `stream_token`, `end_response`)
- [x] Voice settings on first message per context
- [x] Sentence-level flushing for natural prosody
- [x] Barge-in handling via `close_context`
- [x] Keep-alive for context timeout prevention
- [x] Word-level tracking (`words_sent` vs `words_spoken`)
- [x] Response handling with camelCase `contextId` and `is_final`
- [x] Gatekeeper pattern (soft lock, hard interrupt)
- [x] HTTP fallback for backchannels (lower latency)
- [x] BufferManager integration for interruption context

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

### Emotion Detection (Hume AI + SharedAudioBuffer)
- [x] SharedAudioBuffer with context_id generation
- [x] HumeHandler WebSocket connection
- [x] Continuous prediction pattern (zero latency at EndOfTurn)
- [x] Emotion category mapping (48 → 10)
- [x] Emotion-tagged transcript output
- [x] Graceful degradation (neutral fallback when Hume unavailable)
- [x] Circuit breaker protection for Hume
- [x] Integration with state machine and LLM prompts

### Production Resilience
- [x] `resilience.py` module with error classification
- [x] Circuit breaker pattern (5 failures → open, 30s recovery)
- [x] Exponential backoff with jitter (500ms → 5s)
- [x] Graceful degradation for all external services
- [x] Health check methods with auto-recovery
- [x] Timeout enforcement on all async operations
- [x] HumeHandler resilience integration
- [x] TTSOutput resilience integration

---

## 14. Error Handling Considerations

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

## 15. Future Enhancements

### v2 Considerations
- Audio event classification (coughs, laughter) for contextual responses
- Sentiment analysis on transcripts for tone adjustment
- Dynamic timing thresholds based on conversation pace
- Multi-language support with language detection

---

## 16. Configuration Architecture

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

## 17. Edge Cases and State Machine Behavior

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

### Version 1.5 (December 4, 2024)
- **Major Update**: Emotion Detection Layer with Hume AI integration (Section 7)
- **Major Update**: Production Resilience Layer (Section 8)
- **Added**: SharedAudioBuffer for dual-stream audio routing (Deepgram + Hume)
- **Added**: HumeHandler for real-time prosody emotion detection (48 dimensions)
- **Added**: Continuous prediction pattern for zero-latency emotion lookup at EndOfTurn
- **Added**: Emotion category mapping (Hume's 48 → simplified 10 categories)
- **Added**: Emotion-tagged transcript format: `<emotion confidence="X.XX">text</emotion>`
- **Added**: `resilience.py` module with production-grade error handling
- **Added**: Circuit breaker pattern (5 failures → open, 30s timeout → half-open)
- **Added**: Exponential backoff with jitter (500ms → 5s, 10% jitter)
- **Added**: Error classification (transient, throttled, permanent)
- **Added**: Graceful degradation (neutral emotion fallback, silent TTS failure)
- **Added**: Health check methods with auto-recovery for HumeHandler and TTSOutput
- **Updated**: Universal Prompt Format (Section 9) - now includes emotion tags
- **Updated**: Component Integration Diagram - shows ThreadiaVA Engine architecture
- **Updated**: Implementation Checklist - all emotion detection and resilience items complete
- **Renamed**: Sections renumbered to accommodate new sections (7-17)

### Version 1.4 (December 4, 2024)
- **Major Update**: ElevenLabs Multi-Context WebSocket integration (Section 6)
- **Added**: Multi-Context WebSocket architecture for natural-sounding streamed TTS
- **Added**: `context_id` management for prosodic consistency across LLM tokens
- **Added**: Sentence-level flushing at `.!?` boundaries
- **Added**: `voice_settings` on first message per context (per ElevenLabs docs)
- **Added**: Keep-alive method for preventing 20-second context timeout
- **Added**: Barge-in handling via `close_context` for clean interruptions
- **Added**: Response handling for camelCase `contextId` and `is_final`
- **Added**: HTTP fallback for backchannels (lower latency than WebSocket context)
- **Updated**: Section 11 TTS checklist - all items complete
- **Updated**: Word-level tracking integrated with TTS output layer

### Version 1.3 (December 4, 2025)
- **Added**: Word-level TTS tracking (Section 4)
- **Added**: `agent_response_words` and `agent_spoken_words` buffers
- **Added**: `mark_words_spoken()` and `mark_words_spoken_by_count()` TTS API
- **Added**: Word lists in `AgentInterruptedContext` for precise interruption tracking

### Version 1.2 (December 4, 2025)
- **Major Update**: Migrated from manual timing to Deepgram Flux event-driven architecture
- **Updated**: Section 2 (Signal Layer) - Complete Flux documentation with events, guarantees, and configuration
- **Updated**: Section 3 (State Machine) - All transitions now driven by Flux events, not manual timers
- **Updated**: Section 9 (Timing Constants) - Flux-managed vs application constants
- **Updated**: Section 11 (Implementation Checklist) - Reflects Flux-based implementation status
- **Added**: Section 15 (Edge Cases) - 10 edge case scenarios with state machine behavior
- **Added**: Eager LLM optimization documentation (speculative calls on EagerEndOfTurn)
- **Added**: AWS Bedrock support as alternative LLM backend
- **Fixed**: Eager LLM race condition (skip duplicate on EndOfTurn if eager already sent)

### Version 1.1 (December 3, 2025)
- **Added**: Configuration Architecture section (Section 14)
- **Clarified**: Voice options come from frontend payload, not .env file
- **Added**: CallConfig dataclass for per-call dynamic settings
- **Added**: Support for SIP headers and room metadata parsing

### Version 1.0 (December 2025)
- Initial specification document

---

*Document Version: 1.5*
*Last Updated: December 4, 2025*
*Author: ThreadiaVA Engineering*
