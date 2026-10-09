import { invoke } from './tauri.js';

// Starting the daemon from the UI never passes load parameters: the daemon
// reads them from the settings file at spawn time, which is where both pages
// write them. Kept in one place so the Settings and Models pages agree.
export function startDaemon() {
  return invoke('daemon_start', {
    model: null,
    keep_alive_secs: null,
    n_gpu_layers: null,
    n_ctx: null,
  });
}

export function stopDaemon() {
  return invoke('daemon_stop');
}

// Stop, then start. The bind address and access token are read only when the
// process spawns, so a restart is the one way a change to either takes effect.
export async function restartDaemon() {
  await stopDaemon();
  // Stopping reaps the process, but the listening port can lag a beat behind on
  // Windows; without the pause the rebind races the socket that just closed.
  await new Promise((resolve) => setTimeout(resolve, 300));
  await startDaemon();
}
