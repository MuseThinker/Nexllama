const DAEMON = import.meta.env.VITE_DAEMON_URL ?? 'http://127.0.0.1:15656';
const SETTINGS_KEY = 'nexllama.settings';

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
  const res = await fetch(DAEMON + path, {
    headers: headers(),
    signal: timeoutMs > 0 ? AbortSignal.timeout(timeoutMs) : undefined,
    ...opts,
  });
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.json();
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
  base: DAEMON,

  health: () => req('/health'),
  status: () => req('/admin/status'),
  models: () => req('/v1/models'),

  // No timeout: reading a model off disk and filling the KV cache is exactly
  // the work that can take minutes, and aborting it halfway would leave the
  // daemon holding a half-loaded model.
  load: (body) =>
    req('/admin/load', { method: 'POST', body: JSON.stringify(body) }, 0),

  unload: (model) =>
    req('/admin/unload', {
      method: 'POST',
      body: JSON.stringify(model ? { model } : {}),
    }),

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
    const res = await fetch(DAEMON + '/v1/chat/completions', {
      method: 'POST',
      headers: headers(),
      body: JSON.stringify({ messages, stream: true, ...body }),
      signal,
    });
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
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
