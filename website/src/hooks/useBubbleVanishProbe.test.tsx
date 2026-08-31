import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { createRef } from 'react'
import { useBubbleVanishProbe } from './useBubbleVanishProbe'

/**
 * Behavioural contract for the bubble-vanish diagnostic probe (issue #7045):
 *   1. Zero cost when off — with the flag unset, the MutationObserver
 *      constructor must NOT be invoked.
 *   2. On every DROP in mounted [data-display-index] rows, log exactly
 *      '[bubbleProbe] mounted rows dropped' plus an object with exactly the
 *      keys mountedBefore, mountedAfter, storeMessages, displayItems, at.
 *   3. Rises / no-change do not log.
 *
 * happy-dom delivers MutationObserver records asynchronously (microtask), so
 * mutating tests await a tick inside act() to let the callback run. The tests
 * drive the REAL hook (no bespoke re-implementation of its logic).
 */

const FLAG = 'kirocrew_debug_bubble_probe'

function makeScroller(rowCount: number): HTMLDivElement {
  const scroller = document.createElement('div')
  for (let i = 0; i < rowCount; i++) {
    const row = document.createElement('div')
    row.setAttribute('data-display-index', String(i))
    scroller.appendChild(row)
  }
  // Attach to the document so happy-dom's MutationObserver observes it.
  document.body.appendChild(scroller)
  return scroller
}

// Let happy-dom flush its MutationObserver microtask queue.
async function flushObserver(): Promise<void> {
  await act(async () => {
    await Promise.resolve()
    await new Promise(resolve => setTimeout(resolve, 0))
  })
}

beforeEach(() => {
  localStorage.clear()
  document.body.innerHTML = ''
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('useBubbleVanishProbe', () => {
  it('constructs NO MutationObserver when the flag is off (zero cost)', () => {
    // Spy constructor wrapping a minimal MutationObserver stand-in.
    const spy = vi.fn(function () {
      return {
        observe: vi.fn(),
        disconnect: vi.fn(),
        takeRecords: vi.fn(() => []),
      }
    })
    vi.stubGlobal('MutationObserver', spy as unknown as typeof MutationObserver)

    const scroller = makeScroller(3)
    const scrollerRef = createRef<HTMLElement>()
    ;(scrollerRef as { current: HTMLElement }).current = scroller

    renderHook(() =>
      useBubbleVanishProbe({ scrollerRef, storeMessages: 3, displayItems: 3 }),
    )

    // Flag is unset → the constructor must never be invoked.
    expect(spy).not.toHaveBeenCalled()
  })

  it('logs the exact prefix and key set on every drop in mounted rows', async () => {
    localStorage.setItem(FLAG, '1')
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {})

    const scroller = makeScroller(5)
    const scrollerRef = createRef<HTMLElement>()
    ;(scrollerRef as { current: HTMLElement }).current = scroller

    renderHook(() =>
      useBubbleVanishProbe({ scrollerRef, storeMessages: 12, displayItems: 8 }),
    )

    // Remove two rows → mounted count drops 5 → 3.
    await act(async () => {
      scroller.querySelectorAll('[data-display-index]')[4].remove()
      scroller.querySelectorAll('[data-display-index]')[3].remove()
    })
    await flushObserver()

    const dropCalls = logSpy.mock.calls.filter(
      c => c[0] === '[bubbleProbe] mounted rows dropped',
    )
    expect(dropCalls.length).toBeGreaterThanOrEqual(1)

    const [prefix, payload] = dropCalls[dropCalls.length - 1]
    expect(prefix).toBe('[bubbleProbe] mounted rows dropped')
    // Exactly these five keys, no more, no fewer.
    expect(Object.keys(payload as object).sort()).toEqual(
      ['at', 'displayItems', 'mountedAfter', 'mountedBefore', 'storeMessages'].sort(),
    )
    const p = payload as {
      mountedBefore: number
      mountedAfter: number
      storeMessages: number
      displayItems: number
      at: number
    }
    expect(p.mountedAfter).toBeLessThan(p.mountedBefore)
    expect(p.storeMessages).toBe(12)
    expect(p.displayItems).toBe(8)
    expect(typeof p.at).toBe('number')
  })

  it('does not log on a rise, and a later drop uses the raised baseline', async () => {
    localStorage.setItem(FLAG, '1')
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {})

    const scroller = makeScroller(2)
    const scrollerRef = createRef<HTMLElement>()
    ;(scrollerRef as { current: HTMLElement }).current = scroller

    renderHook(() =>
      useBubbleVanishProbe({ scrollerRef, storeMessages: 4, displayItems: 4 }),
    )

    const dropCalls = () =>
      logSpy.mock.calls.filter(c => c[0] === '[bubbleProbe] mounted rows dropped')

    // Rise: 2 → 4. No drop line.
    await act(async () => {
      for (let i = 2; i < 4; i++) {
        const row = document.createElement('div')
        row.setAttribute('data-display-index', String(i))
        scroller.appendChild(row)
      }
    })
    await flushObserver()
    expect(dropCalls()).toHaveLength(0)

    // Drop: 4 → 1. Baseline should be the raised 4, not the original 2.
    await act(async () => {
      const rows = scroller.querySelectorAll('[data-display-index]')
      rows[3].remove()
      rows[2].remove()
      rows[1].remove()
    })
    await flushObserver()

    const calls = dropCalls()
    expect(calls.length).toBeGreaterThanOrEqual(1)
    const payload = calls[calls.length - 1][1] as {
      mountedBefore: number
      mountedAfter: number
    }
    expect(payload.mountedBefore).toBe(4)
    expect(payload.mountedAfter).toBe(1)
  })
})
