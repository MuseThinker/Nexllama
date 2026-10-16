export const hasTauri =
  typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;

export async function invoke(cmd, args) {
  if (!hasTauri) {
    throw new Error('This action requires the desktop app (Tauri).');
  }
  const { invoke: call } = await import('@tauri-apps/api/core');
  return call(cmd, args);
}

export async function listen(event, handler) {
  if (!hasTauri) return () => {};
  const { listen: on } = await import('@tauri-apps/api/event');
  return on(event, handler);
}

export async function pickFolder(title) {
  if (!hasTauri) return null;
  const { open } = await import('@tauri-apps/plugin-dialog');
  const selected = await open({ directory: true, multiple: false, title });
  return selected ?? null;
}

/// Open a URL in the system browser. A webview `<a target="_blank">` does not
/// leave the app, so this goes through the opener plugin on the desktop and
/// falls back to a new tab in the browser dev server.
export async function openExternal(url) {
  if (!url) return;
  if (!hasTauri) {
    window.open(url, '_blank', 'noopener');
    return;
  }
  const { openUrl } = await import('@tauri-apps/plugin-opener');
  await openUrl(url);
}
