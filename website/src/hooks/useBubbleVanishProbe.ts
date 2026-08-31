/**
 * useBubbleVanishProbe — opt-in diagnostic for the "recent bubbles briefly
 * vanish then reappear" symptom (issue #7045).
 *
 * What it does:
 *   When enabled, attaches a MutationObserver to the transcript scroller and,
 *   on EVERY decrease in the number of mounted virtual rows, logs a single
 *   console line:
 *
 *     [bubbleProbe] mounted rows dropped
 *       { mountedBefore, mountedAfter, storeMessages, displayItems, at }
 *
 *   The mounted-row count is `scroller.querySelectorAll('[data-display-index]')`
 *   — one entry per mounted virtual row (see ChatPage's row render sites). The
 *   accompanying `storeMessages` (redux `state.chat.messages.length`) and
 *   `displayItems` (post-grouping display list length) let the reader classify
 *   a drop without re-deriving it:
 *     - storeMessages fell with the rows → store/fetch path;
 *     - displayItems fell while storeMessages held → grouping collapse;
 *     - both held while mounted rows fell → windowing path.
 *
 * Activation (reload semantics):
 *   Set `localStorage.kirocrew_debug_bubble_probe = '1'` and reload. The flag is
 *   read ONCE at hook init and frozen for the hook's lifetime — toggling it
 *   mid-session does not construct or destruct the observer. Reload is the
 *   documented activation path, mirroring how the issue describes turning it on.
 *
 * Zero cost when off:
 *   When the flag is not '1', the effect early-returns BEFORE constructing a
 *   MutationObserver — the constructor itself never runs. This is load-bearing
 *   and pinned by a test (useBubbleVanishProbe.test.tsx).
 *
 * This is a developer console instrument, not user-facing UI, so the log string
 * is intentionally not routed through i18n.
 */
import { useEffect, useRef } from 'react'

// localStorage flag; set to '1' and reload to turn the probe on.
const DEBUG_FLAG_KEY = 'kirocrew_debug_bubble_probe'

// Read the flag once, guarded so SSR / happy-dom without storage cannot throw.
function readEnabledOnce(): boolean {
  try {
    if (typeof localStorage === 'undefined') return false
    return localStorage.getItem(DEBUG_FLAG_KEY) === '1'
  } catch {
    return false
  }
}

export function useBubbleVanishProbe(opts: {
  // Accept both the nullable and non-nullable RefObject shapes that useRef
  // produces (mirrors virtualizer/types.ts externalScrollerRef), so ChatPage's
  // `useRef<HTMLDivElement>(null)` scrollerRef passes without a cast.
  scrollerRef:
    | React.RefObject<HTMLElement | null>
    | React.RefObject<HTMLDivElement | null>
    | React.RefObject<HTMLDivElement>
  storeMessages: number
  displayItems: number
}): void {
  const { scrollerRef, storeMessages, displayItems } = opts

  // Freeze activation for the hook's lifetime: the flag is sampled once on the
  // first render and never re-read, so mid-session toggling is inert (reload is
  // the activation path). useRef's initializer runs exactly once.
  const enabledRef = useRef<boolean | null>(null)
  if (enabledRef.current === null) enabledRef.current = readEnabledOnce()
  const enabled = enabledRef.current

  // Live refs for the counts so the observer callback (stable-identity, created
  // once per attach) always sees the latest values WITHOUT the effect having to
  // re-subscribe when only these numbers change. Mirrors the streamingIndexRef
  // pattern in useVirtualChat.ts.
  const storeMessagesRef = useRef(storeMessages)
  storeMessagesRef.current = storeMessages
  const displayItemsRef = useRef(displayItems)
  displayItemsRef.current = displayItems

  useEffect(() => {
    // ZERO-COST WHEN OFF: bail before constructing anything. No MutationObserver
    // is created when the flag is off.
    if (!enabled) return

    const scroller = scrollerRef.current
    if (!scroller) return // effect re-runs when scrollerRef identity changes
    if (typeof MutationObserver === 'undefined') return

    const countRows = () =>
      scroller.querySelectorAll('[data-display-index]').length

    // Seed the baseline from the current DOM so the first mutation compares
    // against a real measurement rather than zero.
    let prevCount = countRows()

    const observer = new MutationObserver(() => {
      const mountedAfter = countRows()
      const mountedBefore = prevCount
      // Update the baseline on rises too, so the NEXT drop compares against the
      // right high-water mark.
      prevCount = mountedAfter
      if (mountedAfter < mountedBefore) {
        // `at` is a wall-clock timestamp (Date.now(), a number) so log lines can
        // be correlated with other timestamped events in the console.
        console.log('[bubbleProbe] mounted rows dropped', {
          mountedBefore,
          mountedAfter,
          storeMessages: storeMessagesRef.current,
          displayItems: displayItemsRef.current,
          at: Date.now(),
        })
      }
    })

    // subtree so a row mounted/unmounted anywhere under the scroller is caught.
    observer.observe(scroller, { childList: true, subtree: true })

    return () => observer.disconnect()
    // storeMessages/displayItems are intentionally NOT deps — they flow through
    // refs so a changing count does not tear down and rebuild the observer.
  }, [enabled, scrollerRef])
}
