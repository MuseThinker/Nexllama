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
