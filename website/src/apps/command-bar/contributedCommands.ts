/**
 * Commands contributed by installed apps.
 *
 * This is what lets a Command Bar row live OUTSIDE this repository. An app
 * declares `contributes.commands` in its manifest; this module turns that
 * declaration into rows the launcher can rank and run. Nothing here executes app
 * code — a contribution is data, and the host is the only thing that acts on it.
 *
 * **Everything below re-validates what the backend already checks.** That is not
 * belt-and-braces for its own sake: `AppManifest` only grew a typed `contributes`
 * field recently, and an unknown top-level manifest key reaches this dashboard
 * through the manifest's `extra` bucket without passing any schema at all. So an
 * app installed by an older gateway, or one whose `app.json` was edited in place,
 * can put an arbitrary object on this path. Manifest data from a third party is
 * untrusted input; the same posture the overlay resolver takes.
 *
 * A bad declaration is SKIPPED with a warning, never thrown: a malformed app must
 * not be able to take the launcher down (and by extension the Cmd+K gesture) for
 * every other app on the instance.
 *
 * Kept pure and dependency-light so the rules are pinned by unit test rather than
 * by reading the component. Icons stay STRINGS here; resolving one to a glyph is
 * the renderer's job.
 */

/** The subset of `GET /api/apps` this module reads. */
export interface CommandAppRecord {
  name: string
  displayName?: string
  enabled?: boolean
  manifest?: {
    contributes?: {
      commands?: unknown
    }
  }
}

/** The one value a contributed command collects before it can act. */
export interface ContributedArgument {
  placeholder: string
  hint: string
  /**
   * Whether a value is acceptable — a HOST-implemented check, chosen by the
   * manifest's `kind`, never a matcher the manifest supplied itself.
   *
   * This is the field that used to be an app-supplied `RegExp`, and the change is
   * the point rather than a refactor. A regex is a small program, and this one runs
   * against the field on every keystroke on the thread that draws the launcher:
   * `^(a+)+$` and `^(a|aa)+$` are both under ten characters and both exponential,
   * and JavaScript cannot interrupt a synchronous match. Screening the pattern
   * syntactically was tried and abandoned — such a check can only recognize shapes,
   * so each version invites the next hostile pattern it does not cover. Every
   * implementation behind this predicate runs in time proportional to the input no
   * matter what the manifest asks for.
   */
  accept: (value: string) => boolean
  /** App-supplied message shown when the value is not accepted. */
  patternError: string
}

export interface ContributedCommand {
  /**
   * Row id, namespaced by the contributing app.
   *
   * The namespace is what makes a contributed row structurally unable to
   * impersonate a builtin one: builtin ids are `command:*` / `setting:*` / `app:*`,
   * and this shape can only ever collide with another contribution from the same
   * app under the same command id — which the manifest already refuses as a
   * duplicate.
   */
  id: string
  /** `name` of the contributing app, for attribution in the UI. */
  appName: string
  /** `displayName` of the contributing app, or its name. */
  appLabel: string
  title: string
  subtitle: string
  /** Host glyph name; the renderer maps it, and an unknown name falls back. */
  icon: string
  keywords: string[]
  /** Prompt template. Contains `{argument}` exactly when `argument` is set. */
  prompt: string
  /** Send the seeded prompt immediately rather than leaving it in the composer. */
  autoSend: boolean
  argument: ContributedArgument | null
}

/**
 * The placeholder a prompt template uses to interpolate the collected argument.
 *
 * A regex rather than a string constant, for two reasons. It is the only form that
 * substitutes every occurrence in one pass; and a string constant holding
 * `{argument}` reads as untranslated UI copy to the strict i18n rule, which is wrong
 * — this is a structural token in a manifest contract, never rendered.
 */
const ARGUMENT_TOKEN_RE = /\{argument\}/g

/** Whether a prompt template interpolates the collected value. */
function interpolatesArgument(prompt: string): boolean {
  // `test` on a /g regex advances lastIndex, so a fresh copy per call keeps this
  // predicate pure — a shared /g regex answers differently on alternate calls.
  return new RegExp(ARGUMENT_TOKEN_RE.source).test(prompt)
}

/** Same narrow kebab slug the manifest enforces. */
const COMMAND_ID_RE = /^[a-z0-9][a-z0-9-]*$/

/** Caps, all mirroring the manifest's own. */
const MAX_HOSTS = 20
const MAX_PROMPT = 4000
const MAX_TITLE = 120

/** Matchers the HOST implements. The manifest picks one; it cannot supply its own. */
const KINDS = ['url', 'text'] as const

/**
 * Resolve a manifest's `kind` to the host's own check, or `null` if unknown.
 *
 * Every branch runs in time proportional to the input. `url` leans on the runtime's
 * URL parser rather than a pattern — that parser is linear and cannot be made to
 * backtrack, which is the entire reason this indirection exists.
 *
 * An unknown kind is refused rather than defaulting to `text`: a manifest asking for
 * a check this host does not have should not silently get a weaker one, since the
 * value goes on to be spliced into an instruction for an agent with tools.
 */
function matcherFor(kind: string, hosts: string[]): ((value: string) => boolean) | null {
  if (kind === 'text') return value => value.trim().length > 0
  if (kind === 'url') {
    return value => {
      let url: URL
      try {
        url = new URL(value.trim())
      } catch {
        return false
      }
      // Only the two web schemes. `javascript:` and `data:` parse as valid URLs, and
      // this value is shown back to the reader and handed to an agent.
      if (url.protocol !== 'https:' && url.protocol !== 'http:') return false
      if (hosts.length === 0) return true
      const host = url.hostname.toLowerCase()
      // A leading dot means "this domain or any subdomain of it". Without it the match
      // is exact, so `github.com` does not admit `github.com.evil.test`.
      return hosts.some(h => (h.startsWith('.') ? host === h.slice(1) || host.endsWith(h) : host === h))
    }
  }
  return null
}

/**
 * Most commands one app may contribute.
 *
 * A group is display-capped anyway, so this is not about layout — it is about an
 * app being unable to make ranking O(a lot) on every keystroke in the root.
 */
const MAX_COMMANDS_PER_APP = 20

function warnContributionSkipped(appName: string, id: unknown, reason: string): void {
  // eslint-disable-next-line no-console -- a refused contribution is invisible otherwise
  console.warn(
    `[command-bar] app ${JSON.stringify(appName)} contributed command ${JSON.stringify(id)} was skipped: ${reason}`,
  )
}

function str(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

/**
 * Build the argument descriptor, or report why it is unusable.
 *
 * Returns `undefined` for "no argument declared" and `null` for "declared but
 * invalid" — the caller must treat those differently: a command with no argument
 * is a perfectly good command, while one whose argument is broken must not run at
 * all, because its prompt interpolates a value that was never checked.
 */
function readArgument(
  appName: string,
  id: string,
  raw: unknown,
): ContributedArgument | null | undefined {
  if (raw === undefined || raw === null) return undefined
  if (typeof raw !== 'object') {
    warnContributionSkipped(appName, id, 'argument is not an object')
    return null
  }
  const obj = raw as Record<string, unknown>
  if ('pattern' in obj) {
    // An app written against the revision of this contract that accepted its own
    // regex. Refused rather than ignored: `pattern` is an unknown key now, so skipping
    // past it would leave the argument on the default `text` matcher — accepting ANY
    // non-empty string — while the app still declares autoSend and still believes its
    // pattern guards the value.
    warnContributionSkipped(
      appName,
      id,
      `argument.pattern is no longer accepted; declare argument.kind (${KINDS.join(', ')}) instead`,
    )
    return null
  }
  const kind = str(obj.kind) || 'text'
  const hostsRaw = Array.isArray(obj.hosts) ? obj.hosts : []
  const hosts = hostsRaw.map(h => str(h).trim().toLowerCase()).filter(Boolean)
  if (hosts.length > MAX_HOSTS) {
    warnContributionSkipped(appName, id, `argument.hosts exceeds ${MAX_HOSTS} entries`)
    return null
  }
  const accept = matcherFor(kind, hosts)
  if (!accept) {
    warnContributionSkipped(appName, id, `argument.kind must be one of ${KINDS.join(', ')}`)
    return null
  }
  if (hosts.length > 0 && kind !== 'url') {
    warnContributionSkipped(appName, id, "argument.hosts applies only to kind 'url'")
    return null
  }
  return {
    placeholder: str(obj.placeholder),
    hint: str(obj.hint),
    accept,
    patternError: str(obj.patternError),
  }
}

function readCommand(app: CommandAppRecord, raw: unknown): ContributedCommand | null {
  if (typeof raw !== 'object' || raw === null) {
    warnContributionSkipped(app.name, raw, 'entry is not an object')
    return null
  }
  const obj = raw as Record<string, unknown>
  const id = str(obj.id)
  if (!COMMAND_ID_RE.test(id)) {
    warnContributionSkipped(app.name, obj.id, 'id must be lowercase alphanumeric with dashes')
    return null
  }
  const title = str(obj.title)
  if (!title) {
    warnContributionSkipped(app.name, id, 'missing title')
    return null
  }
  if (title.length > MAX_TITLE) {
    warnContributionSkipped(app.name, id, `title exceeds ${MAX_TITLE} characters`)
    return null
  }
  const prompt = str(obj.prompt)
  if (!prompt) {
    warnContributionSkipped(app.name, id, 'missing prompt')
    return null
  }
  if (prompt.length > MAX_PROMPT) {
    warnContributionSkipped(app.name, id, `prompt exceeds ${MAX_PROMPT} characters`)
    return null
  }

  const argument = readArgument(app.name, id, obj.argument)
  if (argument === null) return null
  const interpolates = interpolatesArgument(prompt)
  if (argument === undefined && interpolates) {
    warnContributionSkipped(app.name, id, 'prompt interpolates {argument} but no argument is declared')
    return null
  }
  if (argument !== undefined && !interpolates) {
    // The reader would be asked for a value the command then ignores, and the
    // command would still run — a silently wrong result rather than a visible one.
    warnContributionSkipped(app.name, id, 'declares an argument but the prompt never uses {argument}')
    return null
  }

  const keywords = Array.isArray(obj.keywords)
    ? obj.keywords.filter((k): k is string => typeof k === 'string')
    : []

  return {
    id: `app:${app.name}:${id}`,
    appName: app.name,
    appLabel: app.displayName || app.name,
    title,
    subtitle: str(obj.subtitle),
    icon: str(obj.icon),
    keywords,
    prompt,
    // `autoSend` is only honoured for a command that COLLECTS something, and the
    // clamp lives here rather than at the call site so no consumer can get it wrong.
    //
    // The host's consent mechanism for autoSend is the resolved-prompt preview, and
    // that preview lives in the argument state. A command with no argument never
    // enters it, so honouring autoSend there would send app-authored text to a
    // tool-enabled agent with nothing shown to the reader at all — the exact hole a
    // misleading row in a hostile manifest would aim for. Such a command still runs;
    // it lands in the composer, where the text is visible and one keystroke sends it.
    //
    // The manifest refuses this combination outright, so a validated app never
    // reaches here with it. This clamp is for the app that skipped that check: an
    // unknown manifest key arrives through `extra` having passed no schema at all.
    autoSend: obj.autoSend === true && argument !== undefined,
    argument: argument ?? null,
  }
}

/**
 * Every valid command contributed by the ENABLED installed apps.
 *
 * Disabled apps contribute nothing: the enable state is the reader's switch for
 * the whole app, and a row that still ran from a disabled app would make that
 * switch a lie.
 *
 * No provenance check beyond that, unlike the overlay resolver — and the
 * difference is the point. An overlay REPLACES a host surface, so only a builtin
 * may claim one; a command ADDS a row that the host renders and runs, which is
 * exactly the capability an external app is supposed to have.
 */
export function contributedCommands(apps: readonly CommandAppRecord[]): ContributedCommand[] {
  const out: ContributedCommand[] = []
  for (const app of apps) {
    if (!app.enabled) continue
    const raw = app.manifest?.contributes?.commands
    if (raw === undefined || raw === null) continue
    if (!Array.isArray(raw)) {
      warnContributionSkipped(app.name, '(all)', 'contributes.commands is not an array')
      continue
    }
    // Sliced BEFORE the loop, so the cap bounds the WORK and not just the output. A
    // per-entry `taken` counter that only advanced on success would let a manifest
    // with fifty thousand malformed entries run fifty thousand validations and fifty
    // thousand `console.warn` calls synchronously on the thread that draws the
    // launcher — every one of them rejected, and the dashboard frozen all the same.
    const entries = raw.slice(0, MAX_COMMANDS_PER_APP)
    if (raw.length > MAX_COMMANDS_PER_APP) {
      warnContributionSkipped(app.name, '(rest)', `more than ${MAX_COMMANDS_PER_APP} commands contributed`)
    }
    for (const entry of entries) {
      const cmd = readCommand(app, entry)
      if (!cmd) continue
      out.push(cmd)
    }
  }
  return out
}

/**
 * Build the prompt to seed, with the collected argument spliced in.
 *
 * The value is inserted verbatim: it has already passed the host matcher named by the
 * command's argument kind, which is where the domain of acceptable input is decided.
 * Quoting or escaping here would corrupt the very thing the matcher admitted (a URL,
 * an identifier) for no gain — the recipient is an agent reading prose, not a shell.
 *
 * Verbatim is exactly why the replacement is a FUNCTION and not the string itself:
 * `String.replace` reads `$&`, `$1` and `$$` in a replacement string as references
 * to the match, so a pasted value containing one of those would be silently
 * rewritten into something the reader never typed. A replacer returns the value
 * untouched.
 */
export function resolvePrompt(command: ContributedCommand, argument: string): string {
  if (!command.argument) return command.prompt
  const value = argument.trim()
  return command.prompt.replace(new RegExp(ARGUMENT_TOKEN_RE.source, 'g'), () => value)
}

/** Whether `value` satisfies the command's declared argument kind. */
export function argumentIsValid(command: ContributedCommand, value: string): boolean {
  if (!command.argument) return true
  return command.argument.accept(value.trim())
}
