# Streaming speech-to-text

## Overview

Live speech-to-text for the dashboard composer. The browser streams 16 kHz mono Int16 PCM over a WebSocket and the server relays partial hypotheses, one or more final transcripts, and (when enabled) an auto-submit signal.

Streaming is a property of the PROVIDER rather than of the endpoint: only the providers named in `stt_stream._STREAMING_PROVIDERS` have a partial-result channel.

| `stt.provider` | Where recognition runs | Cost | Precondition |
|---|---|---|---|
| `apple` | the OS, on-device SpeechAnalyzer | free | macOS 26 or later, and a Swift toolchain to build the helper |
| `transcribe` | AWS Transcribe Streaming | billed per audio-second | the `voice` extra, and a recorded AWS consent |

The other providers — `whisper` (the default), `mlx`, `parakeet` and `faster` — transcribe a whole recording at a time, so they have no partial to stream. `whisper`, `mlx` and `parakeet` each need their own out-of-band CLI install; `faster` installs into the gateway's own interpreter, because `POST /api/stt/install` runs `sys.executable -m pip install faster-whisper` and the library is imported in-process. Both routes are covered by [configuration](../../../src/kiro_crew/docs/configuration.md) § Speech-to-text.

The batch path at `POST /api/stt/transcribe` (`transcribe.transcribe_audio`) serves
whole files instead: a Slack voice memo, a channel voice note, an upload, and every
dictation on a provider that cannot stream. Both paths read the one provider setting
and apply the same redaction. `transcribe.filter_hallucinations` additionally runs on
the whisper-family providers' output (`_WHISPER_FAMILY_PROVIDERS`), because whisper
emits caption boilerplate on near-silence and an emptied transcript is reported as
nothing heard rather than written into an agent's notes. It is that recogniser's own
artefact, so it is not applied to `apple`, `transcribe` or `parakeet` — the last runs
an NVIDIA Parakeet model rather than a whisper one, so being a local CLI provider does
not put it in the family.

Compressed input (WebM, M4A, ogg/Opus) is decoded by a **system** FFmpeg, a
prerequisite the user installs rather than something Kiro Crew ships.
`transcribe.ensure_ffmpeg_in_path()` finds it without the user editing `PATH`: it
walks a fixed list of per-user and per-platform install directories in reverse and
**prepends** each one that actually holds an ffmpeg, so the list's own order survives
at the front of `PATH` and a located install outranks whatever was inherited. It has
to reach past `PATH` rather than trust it, because a GUI-launched gateway inherits a
minimal environment that need not contain the user's shell `PATH` and so carries no
Homebrew, winget or scoop prefix. `faster` is the one exception and decodes in-process
through PyAV's bundled FFmpeg.

## Architecture

```
mic -> AudioWorklet (16 kHz mono Int16 PCM) -> WebSocket /api/ws/stt
    -> provider session (apple: on-device SpeechAnalyzer | transcribe: TranscribeStreamingClient)
    -> partial / final / endpoint frames
    -> composer (partial tail replaced in place)
```

### Components

| Component | File | Role |
|---|---|---|
| WS endpoint | `src/kiro_crew/dashboard/stt_stream.py` | One provider session per connection, plus the caps and the SEL audit pair |
| Apple helper | `src/kiro_crew/apple_speech/` | Swift `StreamTranscribe.swift` and `AppleTranscribe.swift`, plus their Python driver |
| Batch providers | `src/kiro_crew/transcribe.py` | `transcribe_audio` and the per-provider runners behind it |
| Config fields | `src/kiro_crew/config/loader.py` | `SttConfig`, and the validation of a stored provider or model |
| Worklet | `website/public/pcm-worklet.js` | Float32-to-16 kHz mono Int16 PCM downsampler |
| Streaming hook | `website/src/hooks/useStreamingStt.ts` | Opens the WS, wires the worklet, emits partial and final |
| Voice hook | `website/src/hooks/useVoiceInput.ts` | Chooses streaming or batch, owns mic and device selection |
| Composer wiring | `website/src/pages/ChatPage.tsx` | Splices the live region into the input box |
| Recording UI | `website/src/components/VoiceDictationPanel.tsx`, `VoiceStatusBar.tsx` | The animated panel, and the thin bar it falls back to |
| Settings UI | `website/src/pages/settings/SttSettings.tsx` | Enable, provider, model, language, and the streaming knobs |

## WebSocket protocol

Client to server:

- Binary frames: raw 16 kHz mono, little-endian Int16 PCM; `test/test_stt_stream.py` pins the transport format and frame limits.
- Text frame `{"type":"stop"}`: the user released the mic. The server finishes
  the utterance and closes, so trailing finals still arrive.

Server to client, JSON. `dashboard.stt_stream` owns the whole wire contract, and both providers emit the same shapes, so the client needs no per-provider branch:

- `{"type":"ready"}`: the session is live and the client may send audio. Capture begins before this arrives, so `useStreamingStt` buffers PCM locally and flushes it in order after readiness. The buffer is capped and drops **oldest-first** so an unavailable server cannot grow browser memory without bound.
- `{"type":"partial","text":"..."}`: an in-progress hypothesis that replaces the
  previous one.
- `{"type":"final","text":"..."}`: the committed transcript for one utterance. A
  session spans many utterances, so `useStreamingStt` accumulates finals rather than
  treating the first as the end.
- `{"type":"endpoint","complete":true}`: a fast background model judged the finished
  segment a complete request, so the composer may submit without a keypress. Only when
  `stt.endpointing` is on.
- `{"type":"error","message":"..."}`: a setup failure, a refusal or a cap. There is no
  machine-readable code on this frame; the English `message` is the whole contract. On
  the `apple` path only the FIRST fatal claimant sends one (`_claim_fatal`), because
  otherwise the duration cap and a concurrent helper failure each emit a frame in the
  window before the other's close lands, and the client shows two contradictory errors
  for a single failure.

Partials and finals both pass `security.redact_credentials` and
`security.redact_exfiltration_urls` before emit. A partial is ephemeral and never
persisted, but it is written into the browser DOM, which makes it an external
surface: a spoken credential must not flash unredacted.

## Activation

The endpoint answers **503** unless all three hold:

1. `stt.enabled`
2. `stt.streaming`
3. `stt.provider` is in `stt_stream._STREAMING_PROVIDERS`

The third is positive membership in a named tuple, never an inequality or a
negation against one provider. Adding a name to that tuple grants it the
endpointer, the caps and the `stt_stream_*` audit identity in one step, so the
grant has to be an explicit edit to the set rather than a side effect of not
matching some other provider. `handlers/core.py` serves the same tuple to the
settings page as `streaming_providers`, so the UI gates its streaming controls on
that capability instead of on a hardcoded name.

After the three gates, each provider has its own precondition and failure frame:

- **apple**: `apple_speech.availability()` decides whether the provider is offered at
  all, and separates "this macOS cannot run it" from "the Swift toolchain is missing",
  because only the second has a fix the user can apply. A helper that then fails to
  start, or stops accepting audio mid-dictation, reports its own reason over the
  `error` frame rather than leaving a live socket that will never transcribe again.
- **transcribe**: `amazon_transcribe` must be importable, and
  `aws_consent.authorize(SERVICE_TRANSCRIBE, profile, region)` must grant.

A provider that cannot stream falls back to batch silently: `useVoiceInput` records
the whole utterance and posts it to `POST /api/stt/transcribe`. The same fallback
covers a browser that cannot stream, since `streamingSupported` requires
`AudioContext`, `AudioWorkletNode`, `WebSocket` and `getUserMedia` and a browser
missing any of them never opens the socket.

### The AWS consent gate is an authorization, not a preference

Transcribe bills per second of audio, so the socket is refused before the client
is constructed and before any audio is read, and the refusal is reported over the
same `error` frame as every other setup failure so the audit pair stays balanced.
The grant is recorded per profile, per region and per resolved account in
`aws_service_consent.json` under the data home, which sits on the read and write
keystone floor, so the agent can neither read the record nor grant itself
permission to spend. The authenticated dashboard is the only writer: there is
deliberately no CLI verb, because a terminal command that records a grant on
request is a grant an automated caller can take.

Moving that check later, adding a CLI verb that records a grant, or reporting the
refusal over some other channel each break one of those three properties.

## Caps and limits

Every limit is a named constant in the module that owns it. The values are not
restated here, because a copied constant goes stale silently.

| Constant | Module | Bounds |
|---|---|---|
| `_MAX_CONCURRENT_SESSIONS` | `dashboard/stt_stream.py` | Sessions per gateway process |
| `_MAX_STREAM_DURATION_SECS` | `dashboard/stt_stream.py` | Wall-clock life of one connection |
| `_MAX_WS_MSG_SIZE` | `dashboard/stt_stream.py` | One inbound audio frame |
| `_MAX_TEXT_FRAME_BYTES` | `dashboard/stt_stream.py` | One inbound control frame |
| `heartbeat` on `WebSocketResponse` | `dashboard/stt_stream.py` | Idle liveness ping interval |

The duration and concurrency caps exist for a different reason per provider and for
an unbounded cost in either case. On `transcribe` an abandoned socket bills per
audio-second and counts against the account's concurrent-stream quota; on `apple` it
holds a helper process and an OS recognition session. The concurrency cap is shared
because both still consume bounded local capacity.

The duration cap is enforced by a dedicated task rather than by an in-loop check,
because `async for msg in ws` only yields on client data and aiohttp answers heartbeat
ping/pong internally: a client that stops sending audio while the socket stays alive
would never evaluate a message-driven deadline, and would hold its session slot until
the gateway restarts.

`test/test_stt_stream.py` pins the transport caps.

## SEL audit pairing: emit before closing

Every accepted connection logs `stt_stream_start`, and **every** exit path must
log a matching `stt_stream_end` (`error`, `refused`, `timeout` or `ok`) or the
audit trail shows an unmatched start. A rejection before the socket is prepared
logs `stt_stream_rejected` instead.

`stt_stream_end` is emitted **before** `await ws.close()`, never after, on the
early-return paths (via `_close_and_end_audit`) and on the normal cleanup path.
`WebSocketResponse.close()` awaits the *peer's* close acknowledgement under its
own timeout, so a client that has already gone away (an abrupt disconnect, a
closed tab) parks the handler inside `close()`, and with the audit after the close
the end event is withheld for as long as that takes. Emitting first makes the
pairing independent of the peer, which is the property a balanced trail actually
needs. The close still runs, is still awaited immediately after, and still
tolerates a broken transport (logged, not raised).

On the `apple` path a claimed fatal cause (`_claim_fatal`) outranks the read loop's
own outcome: the loop can exit cleanly because the cap or the relay closed the socket
under it, and recording that as `ok` would report a session that died as a session
that finished.

Tests asserting on the audit pair must **wait** for the end event: neither
receiving the error frame nor exiting the `TestClient` context orders the
assertion after the server handler's remaining steps, so asserting straight after
either one is a race.

## Frozen-prefix behaviour

`ChatPage.tsx` snapshots the composer's contents and the caret on the first
`partial` of an utterance. Later partials replace only the live region after that
snapshot, so anything the user typed before speaking survives, and the caret does
not jump. The snapshot clears on the final, so the next utterance starts from the
newly committed text.

## Deliberately not built

- **Streaming on the whole-file providers.** `whisper`, `mlx`, `parakeet` and
  `faster` are invoked once per recording and expose no partial-result channel, so a
  live region would have nothing to show until the end.
- **Speaker diarisation and word-level timestamps.** Neither has a consumer in
  the composer, and both change the frame shape every client conforms to.
- **Fan-out of one utterance to several agents.** One session drives one
  composer.
