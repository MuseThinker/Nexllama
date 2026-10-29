import { hasTauri, invoke } from './tauri.js';

/// Ask the updater endpoint whether a newer app version exists.
///
/// The lite installer omits the CUDA runtime, so it must keep receiving lite
/// packages. The manifest carries that as a custom updater target; full builds
/// keep the default OS/architecture target.
export async function checkAppUpdate() {
  if (!hasTauri) return null;
  const { check } = await import('@tauri-apps/plugin-updater');
  const edition = await invoke('app_edition').catch(() => 'full');
  return edition === 'lite' ? check({ target: 'windows-x86_64-lite' }) : check();
}

/// Current version as embedded in the running executable.
export async function appVersion() {
  if (!hasTauri) return '';
  const { getVersion } = await import('@tauri-apps/api/app');
  return getVersion();
}

/// Download and install an update returned by `checkAppUpdate`.
///
/// The Tauri updater verifies the update signature before installing. Windows
/// exits the app while the NSIS installer runs; on other targets we ask the
/// process plugin to relaunch once installation has finished.
export async function installAppUpdate(update, onProgress) {
  if (!update) return;
  let downloaded = 0;
  let total = 0;
  await update.downloadAndInstall((event) => {
    switch (event.event) {
      case 'Started':
        total = event.data.contentLength ?? 0;
        onProgress?.({ downloaded, total });
        break;
      case 'Progress':
        downloaded += event.data.chunkLength ?? 0;
        onProgress?.({ downloaded, total });
        break;
      case 'Finished':
        onProgress?.({ downloaded: total || downloaded, total });
        break;
    }
  });
  try {
    const { relaunch } = await import('@tauri-apps/plugin-process');
    await relaunch();
  } catch {
    // Windows has already exited for the installer; nothing left to do.
  }
}
