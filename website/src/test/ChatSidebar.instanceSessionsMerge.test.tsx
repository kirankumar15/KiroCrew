/**
 * Test: live sessions from connected remote instances MERGE into the Sessions
 * list by recency, rather than being appended after every local row.
 *
 * WHY THIS EXISTS AS A SIDEBAR TEST and not only a hook test: the hook returning
 * correct rows is not the property that broke. `history` arrives date-desc from
 * the backend, so the sidebar SKIPS its sort for the `date-desc` key as an
 * optimisation. Concatenating the hook's rows onto that pre-sorted array is a
 * type-correct change that silently violates the premise of that fast path — the
 * result is two sorted runs, not one — so every remote row rendered BELOW every
 * local row. That is the exact "local list with a remote list stuck on the end"
 * shape this feature exists to replace, and at the bottom of a long list it reads
 * as the feature not working at all. Only a test that asserts RENDERED ORDER
 * across the merge catches it; the hook's own spec passes either way.
 *
 * Mock scaffolding mirrors ChatSidebar.federatedSearch.test.tsx (which mirrors
 * ChatSidebar.offline.test.tsx, the owner of the mock setup).
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { Provider } from 'react-redux'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createTestStore } from './helpers'
import { ThemeProvider } from '../hooks/useTheme'
import { PREVIEW_INSTANCE_SESSIONS } from '../utils/previewFlags'

// Local history rows genuinely carry `modified` in epoch SECONDS. A remote slot
// does NOT: it carries the peer's ISO ladder, which the hook collapses into
// `modified`. Timestamps are NOW-RELATIVE so they land in real date segments:
// all three rows belong to the same bucket, so a correct list prints ONE header.
//
// The remote row's `created` is deliberately 30 days old while its last activity
// is minutes ago. That is the exact shape that produced duplicate `YESTERDAY` /
// `LAST 7 DAYS` headers: ranking read the ladder, but the segment header read
// `modified ?? created`, so an absent `modified` segmented the row by creation and
// the bucket flipped mid-list.
const NOW_S = Math.floor(Date.now() / 1000)
const LOCAL_NEWER = NOW_S - 60
const LOCAL_OLDER = NOW_S - 180

const { instanceChatSlotsMock, listInstancesMock } = vi.hoisted(() => ({
  instanceChatSlotsMock: vi.fn().mockResolvedValue([
    {
      key: 'chat-9',
      title: 'REMOTE middle row',
      // Last activity ~2 min ago: between the two local rows.
      last_turn_ts: new Date(Date.now() - 120_000).toISOString(),
      // Created a month ago — a DIFFERENT date bucket than the activity above.
      created: new Date(Date.now() - 30 * 86_400_000).toISOString(),
      agent: 'default',
    },
  ]),
  listInstancesMock: vi.fn().mockResolvedValue({
    instances: [{ id: 'inst-a', name: 'astro', status: { state: 'connected' } }],
  }),
}))

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>()
  return {
    ...actual,
    api: {
      ...Object.fromEntries(
        [
          'sessions', 'chatSlots', 'chatSlotDetail', 'createChatSlot', 'deleteChatSlot',
          'resumeChatSlot', 'deleteSession', 'agentDetail', 'spawnList', 'fetchHistory',
          'renameSlot', 'forkSession', 'connectInstance', 'sessionsSearch',
          'instancesSearchSessions',
        ].map(k => [k, vi.fn().mockResolvedValue({})]),
      ),
      chatFolders: vi.fn().mockResolvedValue([]),
      listInstances: listInstancesMock,
      instanceChatSlots: instanceChatSlotsMock,
    },
  }
})

// Browser API stubs
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((q: string) => ({
    matches: false, media: q, onchange: null,
    addListener: vi.fn(), removeListener: vi.fn(),
    addEventListener: vi.fn(), removeEventListener: vi.fn(), dispatchEvent: vi.fn(),
  })),
})
globalThis.fetch = vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({}) }) as unknown as typeof fetch

import ChatSidebar from '../pages/ChatSidebar'
import type { ChatSlot, ChatHistoryItem } from '../types'
import type { RootState } from '../store'

const slot = (key: string, title?: string): ChatSlot => ({
  key, title: title ?? key, messages: 1, running: false, mode: '', created: '', last_ts: '2026-01-01T00:00:00Z',
} as ChatSlot)

const histItem = (key: string, title: string, modified: number): ChatHistoryItem => ({
  key, title, modified,
} as unknown as ChatHistoryItem)

function renderSidebar() {
  // LIVE slots carry the ISO ladder, same as a remote row — that is what lets the
  // two interleave. The remote row's last activity sits between these two.
  const slots = [
    { ...slot('s-new', 'LIVE newer slot'), last_turn_ts: new Date(Date.now() - 60_000).toISOString() },
    { ...slot('s-old', 'LIVE older slot'), last_turn_ts: new Date(Date.now() - 180_000).toISOString() },
  ] as ChatSlot[]
  const history = [
    histItem('h-new', 'LOCAL history row', LOCAL_NEWER),
    histItem('h-old', 'LOCAL older history row', LOCAL_OLDER),
  ]
  const store = createTestStore({
    dashboard: {
      status: { platform: 'darwin' },
      connected: true,
      slots,
      approvalMode: 'normal', channelTrusted: false, refreshTrigger: 0, unreadSlots: [], updateProgress: null,
      subagentRunning: {}, subagentDetails: {}, subagentText: {},
      sessionDefaultColor: null, sessionColorsMode: 'tint', sessionColorsPalette: 'horizon', sessionColorsIntensity: 'clear',
      slotsLoaded: true,
    } as unknown as RootState['dashboard'],
    chat: {
      activeSlot: 's1',
      messages: [], slotRunning: false, slotStopping: false, slotState: 'idle',
      slotStatusDetail: {}, slotHasMore: false, slotOldestIndex: 0, loadingOlder: false,
      lastChunkSeq: undefined,
      history, historyHasMore: false, historyOffset: history.length,
      pendingInput: null, slotContextPct: {}, voicePlaying: false, voiceAudio: null,
      subagents: {}, toolLog: [], activityOpen: false, activityTab: 'tools', slotActivity: {}, slotHistory: [],
      slotMessages: {}, slotLoading: false,
    } as unknown as RootState['chat'],
  })
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const view = render(
    <QueryClientProvider client={qc}>
      <Provider store={store}>
        <ThemeProvider>
          <MemoryRouter>
            <ChatSidebar
              slots={slots}
              activeSlot={'s1'}
              unreadSlots={[]}
              history={history}
              historyHasMore={false}
              defaultAgent={'default'}
              installedAgents={[]}
            />
          </MemoryRouter>
        </ThemeProvider>
      </Provider>
    </QueryClientProvider>,
  )
  // History rows live behind the Older Sessions disclosure.
  fireEvent.click(screen.getByRole('button', { name: /^older sessions$/i }))
  return view
}

describe('ChatSidebar – remote instance sessions merge into the list', () => {
  beforeEach(() => {
    instanceChatSlotsMock.mockClear()
    listInstancesMock.mockClear()
    localStorage.clear()
  })

  it('orders a remote row BETWEEN local LIVE sessions by recency', async () => {
    localStorage.setItem(PREVIEW_INSTANCE_SESSIONS, '1')
    const { container } = renderSidebar()

    await waitFor(() => {
      expect(container.textContent).toContain('REMOTE middle row')
    })

    const text = container.textContent ?? ''
    const newer = text.indexOf('LIVE newer slot')
    const remote = text.indexOf('REMOTE middle row')
    const older = text.indexOf('LIVE older slot')

    expect(newer).toBeGreaterThanOrEqual(0)
    expect(older).toBeGreaterThanOrEqual(0)
    // Interleaved by recency among the LIVE sessions — these are the peer's OPEN
    // slots, so they belong with local open sessions, not in the closed-tab drawer.
    expect(newer).toBeLessThan(remote)
    expect(remote).toBeLessThan(older)
  })

  it('prints each date-segment header once, even when a remote row was created in another bucket', async () => {
    localStorage.setItem(PREVIEW_INSTANCE_SESSIONS, '1')
    const { container } = renderSidebar()

    await waitFor(() => {
      expect(container.textContent).toContain('REMOTE middle row')
    })

    // Segment headers are the only uppercase-tracking labels in this list. A
    // correctly ordered list changes bucket monotonically, so no label repeats;
    // a row segmented by a value it did NOT sort by makes the bucket flip and
    // print the same header twice.
    const headers = Array.from(
      container.querySelectorAll('div.uppercase'),
    ).map(el => (el.textContent || '').trim()).filter(Boolean)

    const seen = new Map<string, number>()
    for (const h of headers) seen.set(h, (seen.get(h) ?? 0) + 1)
    const repeated = [...seen.entries()].filter(([, n]) => n > 1)
    expect(repeated).toEqual([])
  })

  it('issues NO remote request and renders no remote row when the flag is off', async () => {
    // Flag absent, i.e. every user who has not opted in.
    const { container } = renderSidebar()

    await waitFor(() => {
      expect(container.textContent).toContain('LIVE newer slot')
    })

    expect(instanceChatSlotsMock).not.toHaveBeenCalled()
    expect(container.textContent).not.toContain('REMOTE middle row')
  })
})
