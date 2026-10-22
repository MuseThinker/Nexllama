const SETTINGS_KEY = 'nexllama.settings';
const FALLBACK_PORT = 15656;

/**
 * True when the page is running inside the desktop app rather than a browser.
 *
 * Read here instead of imported from `tauri.js` so this module stays usable
 * without pulling the Tauri API into a browser bundle.
 */
function inDesktopApp() {
  return typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;
}

/**
 * The daemon's base URL, resolved from the configured port.
 *
 * Read at call time rather than captured once: the port lives in the settings
 * the user edits, and a change takes effect after the daemon restarts — without
 * a page reload the module constant would keep pointing at the old port.
 *
 * In a browser there is nothing to configure. The daemon is serving the page it
 * was asked to serve (`--ui-dir`), so every call is same-origin: an empty base
 * means relative URLs, and the container that ships this exposes one port
 * instead of a second proxy that would have to forward CORS and the token.
 */
function baseUrl() {
  const override = import.meta.env.VITE_DAEMON_URL;
  if (override) return override;
  if (!inDesktopApp()) return '';
  let port = FALLBACK_PORT;
  try {
    const p = Number(JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}')?.port);
    if (Number.isInteger(p) && p > 0 && p < 65536) port = p;
  } catch {
    /* private mode, or settings that are not JSON */
  }
  return `http://127.0.0.1:${port}`;
}

/** The token from Settings, if the user configured one. */
function apiKey() {
  try {
    const key = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}')?.apiKey;
    return typeof key === 'string' && key.trim() ? key.trim() : null;
  } catch {
    return null;
  }
}

function headers(extra = {}) {
  const out = { 'content-type': 'application/json', ...extra };
  const key = apiKey();
  if (key) out.authorization = `Bearer ${key}`;
  return out;
}

/// How long a plain request waits before giving up.
///
/// A daemon busy loading a model does not answer `/admin/status` for a while,
/// and a 5-second poll that never gives up stacks request on request until the
/// browser's six connections per host are all busy. Most calls are quick.
const REQUEST_TIMEOUT_MS = 10_000;

async function req(path, opts = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  let res;
  try {
    res = await fetch(baseUrl() + path, {
      headers: headers(),
      signal: timeoutMs > 0 ? AbortSignal.timeout(timeoutMs) : undefined,
      ...opts,
    });
  } catch (e) {
    throw readableAbort(e, timeoutMs);
  }
  if (!res.ok) throw httpError(res);
  return res.json();
}

/**
 * Turn the browser's own abort into something a person can act on.
 *
 * `AbortSignal.timeout` rejects with a `TimeoutError` whose message is literally
 * "signal timed out" — the browser's words, not ours, and what a slow unload
 * used to show with nothing to explain it. The caller sees a marked error and
 * says it in the user's language; the requests that can legitimately take
 * minutes (load, unload) pass `timeoutMs = 0` and never arrive here.
 */
function readableAbort(e, timeoutMs) {
  if (e && (e.name === 'TimeoutError' || e.name === 'AbortError')) {
    const err = new Error('the daemon did not answer in time');
    err.timedOut = true;
    err.timeoutSecs = Math.round(timeoutMs / 1000);
    return err;
  }
  return e;
}

/**
 * An error that carries the status code.
 *
 * Callers need to tell "the token is wrong" from "the daemon is not running" —
 * only one of those is something the user fixes by pasting a token — and a
 * message string is the wrong place to look for it.
 *
 * A 401 also raises `nexllama:unauthorized` on the window. A token protects
 * every route except `/health`, so nothing on screen would otherwise ever say
 * why the app is empty: the status dot is green (the daemon answered) and every
 * panel failed. One event from here is enough for the shell to say so once,
 * rather than each page catching its own errors to repeat it.
 */
function httpError(res) {
  const err = new Error(`${res.status}: ${res.statusText}`);
  err.status = res.status;
  if (res.status === 401 && typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('nexllama:unauthorized'));
  }
  return err;
}

/**
 * One SSE frame: an optional `event:` line plus its `data:` payload.
 *
 * The event name matters: the daemon reports failures as `event: error` with a
 * plain-text body, so reading only `data:` and `JSON.parse`-ing it (with the
 * throw swallowed) turned every server-side error into silence — a reply that
 * simply stopped, with nothing on screen to say why.
 */
function parseEvent(frame) {
  let event = 'message';
  const data = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) data.push(line.slice(5).trim());
  }
  const payload = data.join('\n');
  if (!payload) return null;
  // OpenAI's terminator, now sent at the end of every stream.
  if (payload === '[DONE]') return { event: 'done' };

  try {
    const parsed = JSON.parse(payload);
    if (parsed && typeof parsed === 'object') return { ...parsed, event };
  } catch {
    /* a human-readable error frame, handled below */
  }
  return { event, error: payload };
}

export const daemon = {
  get base() {
    return baseUrl();
  },

  health: () => req('/health'),
  status: () => req('/admin/status'),
  models: () => req('/v1/models'),

  /**
   * The models folder, as the daemon sees it.
   *
   * This is what the app's own library command returns on the desktop. Serving
   * it over HTTP is what lets a browser show the same library: the rows are
   * built by `model-index`, the same code the app calls, so the two cannot
   * disagree about which files are one model.
   *
   * A daemon without `--models-dir` answers `dir: null` and no models, which is
   * a fact about the server rather than a failure — the page shows an empty
   * library and says there is no folder configured.
   */
  localModels: () => req('/admin/models'),

  // HuggingFace catalog. Search and listing talk to the Hub, so they get longer
  // than the default: a cold upstream connection can spend several seconds
  // before it answers at all.
  hubAuthors: () => req('/admin/hub/authors'),
  hubSearch: (query, author = null, limit = 20) => {
    const params = new URLSearchParams({ q: query, limit: String(limit) });
    if (author) params.set('author', author);
    return req(`/admin/hub/search?${params}`, {}, 30_000);
  },
  hubFiles: (repo) =>
    req(`/admin/hub/files?${new URLSearchParams({ repo })}`, {}, 30_000),
  hubDownload: (repo, filename) =>
    req(
      '/admin/hub/download',
      { method: 'POST', body: JSON.stringify({ repo, filename }) },
      0,
    ),
  hubDownloadStatus: (repo, filename) =>
    req(`/admin/hub/download?${new URLSearchParams({ repo, filename })}`),
  hubPause: (repo, filename) =>
    req('/admin/hub/download/pause', {
      method: 'POST',
      body: JSON.stringify({ repo, filename }),
    }),
  hubCancel: (repo, filename) =>
    req('/admin/hub/download/cancel', {
      method: 'POST',
      body: JSON.stringify({ repo, filename }),
    }),

  // No timeout: reading a model off disk and filling the KV cache is exactly
  // the work that can take minutes, and aborting it halfway would leave the
  // daemon holding a half-loaded model.
  load: (body) =>
    req('/admin/load', { method: 'POST', body: JSON.stringify(body) }, 0),

  // Default timeout on purpose. Measuring reads real metadata, but it is a
  // header read, not a load — and the download precheck runs *before* the
  // transfer, so a hung daemon here would hold the download forever with
  // nothing on screen to say why. Timing out falls back to "start anyway".
  //
  // Generous rather than the usual ten seconds: this measurement loads the
  // checkpoint to weigh it against the device, and a 27B off a slow disk takes
  // far longer than ten. Reading the report is worth waiting for; a check that
  // gives up early answers "start anyway" for every large model.
  fit: (body) =>
    req('/admin/fit', { method: 'POST', body: JSON.stringify(body) }, 120_000),

  // No timeout: freeing a model is the same kind of work as loading one. A 27B
  // dropping its KV cache and weights can take longer than ten seconds, and
  // aborting mid-unload used to surface as `500: TimeoutError: signal timed out`
  // for an unload that had in fact succeeded.
  unload: (model) =>
    req(
      '/admin/unload',
      {
        method: 'POST',
        body: JSON.stringify(model ? { model } : {}),
      },
      0,
    ),

  chat: (messages, opts = {}) =>
    req('/v1/chat/completions', {
      method: 'POST',
      body: JSON.stringify({ messages, stream: false, ...opts }),
    }),

  /**
   * Stream one completion, yielding a parsed frame per event.
   *
   * `signal` aborts the request, and the reader is released on the way out
   * either way. Dropping out of the loop alone is not enough: it leaves the
   * response body open, and the daemon goes on generating tokens into a socket
   * nobody is reading — which also holds the model's worker thread, so the next
   * request waits for a reply that was abandoned.
   */
  async *stream(messages, opts = {}) {
    const { signal, ...body } = opts;
    const res = await fetch(baseUrl() + '/v1/chat/completions', {
      method: 'POST',
      headers: headers(),
      body: JSON.stringify({ messages, stream: true, ...body }),
      signal,
    });
    if (!res.ok) throw httpError(res);
    if (!res.body) throw new Error(`${res.status}: the daemon sent no response body`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const frames = buffer.split('\n\n');
        buffer = frames.pop() ?? '';

        for (const frame of frames) {
          const ev = parseEvent(frame);
          if (ev) yield ev;
        }
      }
    } finally {
      await reader.cancel().catch(() => {});
    }
  },
};
