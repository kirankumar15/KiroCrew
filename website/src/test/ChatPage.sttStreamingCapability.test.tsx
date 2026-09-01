// The persisted `stt.streaming` flag is per-install and can outlive the provider
// that supported it: a config written while the provider streamed keeps
// `streaming: true` after the provider is replaced (including by the loader's
// own degrade of a value no longer in the enum). ChatPage must therefore gate
// the flag on the backend's served capability list rather than trust it, because
// the streaming branch opens a socket the gateway refuses for a non-streaming
// provider and returns without falling back to batch — a dead mic with no
// in-product recovery, since the Settings toggle is hidden for that provider.
import { describe, it, expect, vi, beforeEach } from 'vitest'
import type { TranscriptOrigin } from '../hooks/useVoiceInput'
import { render, screen, act, waitFor } from '@testing-library/react'
import { Provider } from 'react-redux'
import { MemoryRouter } from 'react-router-dom'
import { configureStore } from '@reduxjs/toolkit'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider } from '../hooks/useTheme'
import chatReducer from '../store/chatSlice'
import dashboardReducer from '../store/dashboardSlice'
import notificationsReducer from '../store/notificationsSlice'
import type { RootState } from '../store'

vi.mock('react-virtuoso', () => ({ Virtuoso: ({ data, itemContent }: { data?: unknown[]; itemContent: (i: number, d: unknown) => React.ReactNode }) => <div data-testid="virtuoso">{data?.map((d: unknown, i: number) => <div key={i}>{itemContent(i, d)}</div>)}</div> }))
vi.mock('../api/client', () => ({
  api: {
    chatSlots: vi.fn().mockResolvedValue([]),
    chatSlotDetail: vi.fn().mockResolvedValue({ messages: [{ role: 'assistant', content: 'hi', cls: '' }], running: false, has_more: false, total: 1 }),
    sendChat: vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({ ok: true }) }),
    chatHistory: vi.fn().mockResolvedValue({ sessions: [] }),
    models: vi.fn().mockResolvedValue([]),
    agents: vi.fn().mockResolvedValue([]),
    agentDetail: vi.fn().mockResolvedValue({}),
    workspaces: vi.fn().mockResolvedValue({ workspaces: [] }),
    slackChannels: vi.fn().mockResolvedValue([]),
    spawnList: vi.fn().mockResolvedValue({ agents: [] }),
    uploadFiles: vi.fn().mockResolvedValue({ paths: [] }),
    screenshot: vi.fn().mockResolvedValue({ path: null }),
    sttConfig: vi.fn(),
  },
  SEARCH_MIN_CHARS: 2,
}))
// Records the `streaming` option ChatPage hands the hook — the value that decides
// which capture path a mic press takes.
const voice = vi.hoisted(() => ({ streaming: null as boolean | null }))
vi.mock('../hooks/useVoiceInput', () => ({
  useVoiceInput: (_onText: (t: string, sessionId: string | null, origin: TranscriptOrigin) => void, opts?: { streaming?: boolean }) => {
    voice.streaming = !!opts?.streaming
    return {
      recording: false,
      transcribing: false,
      sessionOwner: null,
      streamEnabled: !!opts?.streaming,
      toggle: vi.fn(),
      start: vi.fn(),
      stop: vi.fn(),
      cancel: vi.fn(),
      prewarm: vi.fn(),
      error: null,
      level: 0,
      deviceLabel: '',
      clearError: vi.fn(),
      partial: '',
      sampleRef: { current: { level: 0, centroid: 0.5, onset: 0 } },
    }
  },
  voiceInputSupported: true,
}))
vi.mock('../hooks/useBranding', () => ({ useBranding: () => ({ botName: 'Test', avatar: '' }) }))
vi.mock('../hooks/useAgents', () => ({ useAgents: () => ({ agents: [], defaultAgent: 'default' }) }))
vi.mock('../components/MarkdownRenderer', () => ({ default: ({ content }: { content: string }) => <span>{content}</span> }))
vi.mock('../components/WelcomeView', () => ({ default: () => null }))
vi.mock('../components/MarkdownPanel', () => ({ default: () => null }))
vi.mock('../pages/chat/ActivityViewer', () => ({ default: () => null }))
vi.mock('../components/DetailPanel', () => ({ default: () => null }))
vi.mock('../hooks/useWebSocket', () => ({ useWebSocket: () => ({ subscribeLogs: () => {} }) }))

Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockReturnValue({ matches: false, addEventListener: vi.fn(), removeEventListener: vi.fn() }),
})

import ChatPage from '../pages/ChatPage'
import { api } from '../api/client'

function makeStore() {
  return configureStore({
    reducer: { dashboard: dashboardReducer, chat: chatReducer, notifications: notificationsReducer },
    preloadedState: {
      dashboard: {
        status: null, connected: true,
        slots: [{ key: 'chat-main', messages: 1, running: false, mode: '', pending_approval: false, waiting_for_input: false, last_activity_ts: undefined }],
        unreadSlots: [], refreshTrigger: 0, approvalMode: 'normal',
        subagentRunning: {}, subagentDetails: {}, subagentText: {},
      } as unknown as RootState['dashboard'],
      chat: {
        activeSlot: 'chat-main', messages: [{ role: 'assistant', content: 'hi', cls: '' }],
        slotRunning: false, slotStopping: false, slotState: 'idle',
        history: [], historyHasMore: false, pendingInput: null,
        subagents: {}, toolLog: [], activityOpen: false, activityTab: 'tools',
        slotHasMore: false, slotOldestIndex: 0, loadingOlder: false,
        slotStatusDetail: {}, slotContextPct: {}, slotActivity: {}, slotHistory: [],
        historyOffset: 0, _wsChunkedDuringFetch: false,
        slotMessages: {}, slotLoading: false,
      } as unknown as RootState['chat'],
      notifications: { items: [] } as unknown as RootState['notifications'],
    },
  })
}

async function mount(stt: Record<string, unknown>) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  // Seed the cache so the FIRST render already carries the config. Waiting for a
  // settled value instead cannot work here: with `sttCfg` still undefined the first
  // render writes `streaming = false`, which is the very value the decisive case
  // asserts — so a guard on that value latches on the pre-settle render and the test
  // passes even with the production gate removed.
  qc.setQueryData(['sttConfig'], { enabled: true, available: true, dictation_panel: true, ...stt })
  await act(async () => {
    render(
      <QueryClientProvider client={qc}>
        <Provider store={makeStore()}>
          <ThemeProvider>
            <MemoryRouter><ChatPage /></MemoryRouter>
          </ThemeProvider>
        </Provider>
      </QueryClientProvider>,
    )
  })
  await waitFor(() => expect(screen.getByLabelText('Message input')).toBeTruthy())
}

const setStt = (stt: Record<string, unknown>) => vi.mocked(api.sttConfig).mockResolvedValue({
  enabled: true, available: true, dictation_panel: true, ...stt,
} as unknown as Awaited<ReturnType<typeof api.sttConfig>>)

beforeEach(() => {
  sessionStorage.clear()
  localStorage.clear()
  voice.streaming = null
})

describe('ChatPage — stt.streaming is gated on the served capability', () => {
  it('ignores a stored streaming flag for a provider that cannot stream', async () => {
    // The upgrade shape: a config written for a streaming provider that the
    // loader has since degraded to `whisper`, with `streaming: true` intact.
    const stt = { provider: 'whisper', streaming: true, streaming_providers: ['transcribe', 'apple'] }
    setStt(stt)
    await mount(stt)
    await waitFor(() => expect(voice.streaming).toBe(false))
  })

  it('honours the flag for a provider the backend lists as streaming', async () => {
    const stt = { provider: 'apple', streaming: true, streaming_providers: ['transcribe', 'apple'] }
    setStt(stt)
    await mount(stt)
    await waitFor(() => expect(voice.streaming).toBe(true))
  })

  it('keeps streaming when an older gateway serves no capability list', async () => {
    // No `streaming_providers` in the payload: fall back to the same default
    // SttSettings uses, so the toggle the user can see and the path the mic takes
    // agree on one answer.
    const stt = { provider: 'transcribe', streaming: true }
    setStt(stt)
    await mount(stt)
    await waitFor(() => expect(voice.streaming).toBe(true))
  })
})
