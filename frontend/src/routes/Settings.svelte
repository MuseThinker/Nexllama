<script>
import { onMount } from 'svelte';
import { invoke, hasTauri, listen } from '../lib/tauri.js';
import { t, locale } from '../lib/i18n.js';
import { theme } from '../lib/theme.js';
import { restartDaemon as daemonRestart } from '../lib/daemonctl.js';

const KEY = 'nexllama.settings';

const defaults = {
  keepAliveSecs: 0,
  backend: 'auto',
  autoStartDaemon: true,
  // `modelsDir` and the load parameters below are edited on the Models page, but
  // they still have to ride along in this object: `set_settings` replaces the
  // whole settings file, so a key this page forgets to carry is deleted from it.
  modelsDir: '',
  nGpuLayers: 999,
  // 0 = size the context to the machine; see the Models page's context field.
  nCtx: 0,
  nBatch: 2048,
  nUbatch: 512,
  mainGpu: 0,
  threads: 0,
  threadsBatch: 0,
  loadMode: 'auto',
  useMlock: false,
  flashAttn: 'auto',
  offloadKqv: true,
  ropeFreqBase: 0,
  ropeFreqScale: 0,
  updateManifestUrl: '',
  updatePublicKey: '',
  bindHost: '127.0.0.1',
  // The port the daemon listens on. Changing it needs a daemon restart, which
  // the Restart button in the Network card performs.
  port: 15656,
  apiKey: '',
};

function local() {
  try {
    return { ...defaults, ...JSON.parse(localStorage.getItem(KEY) ?? '{}') };
  } catch {
    return { ...defaults };
  }
}

let settings = $state(local());
let saved = $state(false);
let error = $state('');
let daemonInfo = $state({ running: false, binary: null, port: 15656, backend: 'cpu' });
let daemonBusy = $state(false);
let backendInfo = $state(null);
let autoStart = $state(local().autoStartDaemon ?? true);
let backendUpdate = $state({ current: '', manifest_url: null, available: [] });
let updateInfo = $state(null);
let checking = $state(false);
let updating = $state('');
let updateProgress = $state(null);
let updateMsg = $state('');

function fmtBytes(n) {
  if (n == null) return '';
  const mb = n / 1048576;
  return mb >= 1024 ? `${(mb / 1024).toFixed(2)} GB` : `${mb.toFixed(1)} MB`;
}
let copied = $state(false);

// Only loopback addresses are safe without a token; anything else is reachable
// from the network, so the UI has to say so out loud.
const exposed = $derived(
  settings.bindHost !== '127.0.0.1' && settings.bindHost !== 'localhost' && settings.bindHost !== '',
);

// Set when the user tried to listen on every interface without a token. The
// choice is refused (the daemon would refuse to start anyway) and the prompt to
// generate one is shown.
let bindBlocked = $state(false);

/// Change the bind address, refusing an exposed one until a token exists.
///
/// The daemon refuses to start on a routable address with no token, so allowing
/// the choice would leave the app with a service that never comes up. Keep
/// loopback and ask for a token instead.
function onBindHostChange(e) {
  const value = e.target.value;
  if (value !== '127.0.0.1' && value !== 'localhost' && !settings.apiKey.trim()) {
    bindBlocked = true;
    settings.bindHost = '127.0.0.1';
    e.target.value = '127.0.0.1';
    return;
  }
  bindBlocked = false;
  settings.bindHost = value;
}

/// A fresh 24-byte token as hex.
///
/// `crypto.getRandomValues` is a browser API, so this works in the desktop shell
/// and in a plain browser alike — no reason to gate it on Tauri.
function generateKey() {
  const bytes = new Uint8Array(24);
  crypto.getRandomValues(bytes);
  settings.apiKey = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
  // A token now exists, so the earlier refusal no longer applies — the user can
  // pick 0.0.0.0 next.
  bindBlocked = false;
}

async function copyKey() {
  const key = settings.apiKey.trim();
  if (!key) return;
  try {
    await navigator.clipboard.writeText(key);
  } catch {
    // Clipboard access can be denied outside a secure context; fall back to the
    // old selection trick rather than leaving the user with nothing.
    const scratch = document.createElement('textarea');
    scratch.value = key;
    scratch.style.position = 'fixed';
    scratch.style.opacity = '0';
    document.body.appendChild(scratch);
    scratch.select();
    document.execCommand('copy');
    scratch.remove();
  }
  copied = true;
  setTimeout(() => (copied = false), 1500);
}

function clearKey() {
  settings.apiKey = '';
  // Without a token, an exposed bind cannot start; fall back to loopback and
  // say why rather than leaving a setting the daemon will refuse.
  if (exposed) {
    settings.bindHost = '127.0.0.1';
    bindBlocked = true;
  }
}

async function refreshBackendUpdate() {
  if (!hasTauri) return;
  try {
    backendUpdate = await invoke('backend_update_status');
  } catch {
    /* ignore */
  }
}

// The keys this page owns and edits. `set_settings` replaces the whole file, so
// anything sent that this page does not edit — the models folder chosen on the
// Models page, the cloud providers, the routing switch, the remembered models —
// has to come from the file as it stands now, never from the snapshot this page
// loaded when it opened. Sending that snapshot is how a user's models folder
// silently reverts to an earlier value.
const OWNED = [
  'keepAliveSecs',
  'backend',
  'nGpuLayers',
  'nCtx',
  'nBatch',
  'nUbatch',
  'mainGpu',
  'threads',
  'threadsBatch',
  'loadMode',
  'useMlock',
  'flashAttn',
  'offloadKqv',
  'ropeFreqBase',
  'ropeFreqScale',
  'updateManifestUrl',
  'updatePublicKey',
  'bindHost',
  'port',
  'apiKey',
];

/// This page's own fields, layered onto whatever is on disk right now.
async function settingsPayload() {
  let base = {};
  if (hasTauri) {
    try {
      base = (await invoke('get_settings')) ?? {};
    } catch {
      /* fall through to this page's values alone */
    }
  }
  const payload = { ...base };
  for (const key of OWNED) {
    if (key in settings) payload[key] = settings[key];
  }
  payload.autoStartDaemon = autoStart;
  localStorage.setItem(KEY, JSON.stringify(payload));
  return payload;
}

async function checkUpdate() {
  checking = true;
  error = '';
  updateMsg = '';
  try {
    await invoke('set_settings', { settings: await settingsPayload() });
    updateInfo = await invoke('check_backend_update');
  } catch (e) {
    error = String(e);
  } finally {
    checking = false;
  }
}

async function applyUpdate(backend) {
  updating = backend;
  error = '';
  updateMsg = '';
  updateProgress = null;
  try {
    const r = await invoke('apply_backend_update', { backend });
    updateMsg = `${r.backend} → ${r.version}`;
    await refreshBackendUpdate();
    await refreshStatus();
    await refreshBackend();
  } catch (e) {
    error = String(e);
  } finally {
    updating = '';
    updateProgress = null;
  }
}

/// One-click install of the GPU backend this machine wants. Persists the
/// manifest URL/key first, since the download reads them from the settings file.
async function downloadGpu(backend) {
  try {
    await invoke('set_settings', { settings: await settingsPayload() });
  } catch {
    /* the apply step reports a missing manifest clearly enough */
  }
  await applyUpdate(backend);
}

async function refreshBackend() {
  if (!hasTauri) return;
  try {
    backendInfo = await invoke('backend_info');
  } catch (e) {
    error = String(e);
  }
}

async function load() {
  if (hasTauri) {
    try {
      const s = await invoke('get_settings');
      if (s && Object.keys(s).length > 0) settings = { ...defaults, ...s };
    } catch (e) {
      error = String(e);
    }
  }
  autoStart = settings.autoStartDaemon;
  if (hasTauri) await readAutostartState();
  refreshStatus();
  refreshBackend();
  refreshBackendUpdate();
}

/**
 * Whether the OS is actually set to launch this app, which is not the same as
 * what the settings file says.
 *
 * The installer offers "start at login" on its options page and writes the
 * registry entry itself, so a fresh install has the entry while the settings
 * file — which only the checkbox writes — still says off. Showing the stored
 * value there would tell the user launch-at-login is off when they had just
 * asked for it, and the first thing they did next would be to turn it on again.
 *
 * Asked of the plugin rather than kept locally, because the plugin is what
 * writes the entry; a registry value it does not recognize is a state the
 * checkbox would happily overwrite.
 *
 * Marked so the auto-save effect below knows this is a load, not an edit: writing
 * the value back on open would be harmless but would mean every visit to this
 * page rewrote the settings file and the registry key.
 */
let syncingAutostart = false;
async function readAutostartState() {
  syncingAutostart = true;
  try {
    autoStart = await invoke('plugin:autostart|is_enabled');
    // The settings file is the app's memory of the same fact; keeping it in step
    // means the next open is right without asking the plugin again.
    settings = { ...settings, autoStartDaemon: autoStart };
  } catch {
    // No plugin, or the call is not allowed in this build: the stored setting is
    // then the best answer available, which is what this fell back to anyway.
  } finally {
    syncingAutostart = false;
  }
}

async function refreshStatus() {
  try {
    if (hasTauri) daemonInfo = await invoke('daemon_status');
  } catch (e) {
    error = String(e);
  }
}

async function persist() {
  saved = false;
  error = '';
  // Build the payload instead of mutating `settings`: the auto-save effect below
  // watches that object, and writing to it here would retrigger itself forever.
  const payload = await settingsPayload();
  if (!hasTauri) {
    flashSaved();
    return;
  }
  try {
    await invoke('set_settings', { settings: payload });
    await invoke('set_autostart', { enabled: autoStart });
    flashSaved();
  } catch (e) {
    error = String(e);
  }
}

function flashSaved() {
  saved = true;
  setTimeout(() => (saved = false), 1500);
}

// The page used to save only when a folder was picked, so typing an API key and
// pressing "generate" changed nothing but the screen. Watching the object covers
// every field on the page, debounced so a keystroke is not a write.
let saveTimer;
let hydrated = false;
$effect(() => {
  // Touch the watched fields so the effect subscribes to them.
  const snapshot = JSON.stringify({ ...settings, autoStartDaemon: autoStart });
  if (!hydrated) {
    // The first run is the load, not an edit.
    hydrated = true;
    return;
  }
  // Reading the launch-at-login entry back changed these fields; that is the
  // installer being told what it already decided, not a user edit to save.
  if (syncingAutostart) return;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => persist(), 600);
});

// The bind address and access token only reach the daemon when it spawns, so
// changing either needs a restart. The button sits in the card that makes the
// change rather than sending the user to the Models page for the same action.
async function restartDaemon() {
  daemonBusy = true;
  error = '';
  try {
    await daemonRestart();
    await refreshStatus();
  } catch (e) {
    error = String(e);
  } finally {
    daemonBusy = false;
  }
}

onMount(() => {
  load();
  if (!hasTauri) return;
  let un;
  listen('update://progress', (e) => (updateProgress = e.payload)).then((f) => (un = f));
  return () => un?.();
});
</script>

<div class="page settings">
  <header>
    <h2>{$t('settings.title')}</h2>
  </header>

  <section class="card">
    <h3>{$t('settings.appearance')}</h3>
    <div class="field">
      <span>{$t('ui.theme')}</span>
      <div class="menu-row">
        <button class:active={$theme === 'light'} onclick={() => theme.set('light')}>{$t('ui.themeLight')}</button>
        <button class:active={$theme === 'dark'} onclick={() => theme.set('dark')}>{$t('ui.themeDark')}</button>
        <button class:active={$theme === 'auto'} onclick={() => theme.set('auto')}>{$t('ui.themeAuto')}</button>
      </div>
    </div>
    <div class="field">
      <span>{$t('ui.language')}</span>
      <div class="menu-row">
        <button class:active={$locale === 'zh'} onclick={() => locale.set('zh')}>中文</button>
        <button class:active={$locale === 'en'} onclick={() => locale.set('en')}>English</button>
      </div>
    </div>
  </section>

  <section class="card">
    <h3>{$t('settings.network')}</h3>
    <label class="field">
      <span>{$t('settings.bindHost')}</span>
      <select value={settings.bindHost} onchange={onBindHostChange}>
        <option value="127.0.0.1">{$t('settings.bindLocal')}</option>
        <option value="0.0.0.0">{$t('settings.bindAll')}</option>
      </select>
    </label>
    <label class="field">
      <span>{$t('settings.port')}</span>
      <input type="number" min="1" max="65535" bind:value={settings.port} />
    </label>
    <label class="field">
      <span>{$t('settings.apiKey')}</span>
      <span class="keyrow">
        <input
          type="text"
          bind:value={settings.apiKey}
          placeholder={$t('settings.apiKeyNone')}
          autocomplete="off"
          spellcheck="false"
        />
        <button class="ghost" onclick={generateKey}>{$t('settings.generate')}</button>
        <button class="ghost" onclick={copyKey} disabled={!settings.apiKey.trim()}>
          {copied ? $t('settings.copied') : $t('settings.copy')}
        </button>
        <button class="ghost" onclick={clearKey} disabled={!settings.apiKey.trim()}>
          {$t('settings.clear')}
        </button>
      </span>
    </label>
    {#if settings.apiKey.trim()}
      <p class="muted">{$t('settings.apiKeyHint')}</p>
    {/if}
    {#if bindBlocked}
      <p class="warn">{$t('settings.bindNeedsKey')}</p>
    {/if}
    {#if exposed}
      {#if !settings.apiKey.trim()}
        <p class="warn">{$t('settings.exposedNoKey')}</p>
      {:else}
        <p class="muted">
          {$t('settings.exposedKey')}
          <code>http://&lt;本机局域网 IP&gt;:{daemonInfo.port}/v1/chat/completions</code>
        </p>
      {/if}
    {/if}
    <div class="restart-row">
      <p class="muted">{$t('settings.networkRestart')}</p>
      <button
        class="ghost"
        onclick={restartDaemon}
        disabled={daemonBusy || !hasTauri}
        title={$t('settings.restartHint')}
      >
        {$t('settings.restart')}
      </button>
    </div>
  </section>

  <section class="card">
    <h3>{$t('settings.lifecycle')}</h3>
    <label class="field">
      <span>{$t('settings.keepAlive')}</span>
      <input type="number" min="0" bind:value={settings.keepAliveSecs} />
    </label>
    <p class="muted">{$t('settings.keepAliveHint')}</p>
    <label class="checkline">
      <input type="checkbox" bind:checked={autoStart} />
      <span>{$t('settings.autostart')}</span>
    </label>
  </section>

  <section class="card">
    <h3>{$t('settings.inference')}</h3>
    <label class="field">
      <span>{$t('settings.backend')}</span>
      <select bind:value={settings.backend}>
        <option value="auto">{$t('settings.backendAuto')}</option>
        <option value="cuda">{$t('settings.backendCuda')}</option>
        <option value="vulkan">{$t('settings.backendVulkan')}</option>
        <option value="cpu">{$t('settings.backendCpu')}</option>
      </select>
    </label>
    {#if backendInfo}
      {@const gpu = backendInfo.gpu_backend}
      {@const installed = backendInfo.available ?? []}
      <p class="muted">
        {$t('settings.gpu')}:
        {backendInfo.gpus?.length ? backendInfo.gpus.join(', ') : $t('settings.gpuNone')}
      </p>
      <p class="muted">
        {$t('settings.runningBackend')}: <code>{daemonInfo.backend}</code>
        {#if gpu} · {$t('settings.gpuSuggested')}: <code>{gpu}</code>{/if}
        · {$t('settings.bundled')}: <code>{installed.join(', ') || '-'}</code>
      </p>
      {#if gpu}
        {#if installed.includes(gpu)}
          <p class="ok">{$t('settings.gpuReady', { gpu })}</p>
        {:else if backendUpdate.manifest_url}
          <div class="row">
            <button onclick={() => downloadGpu(gpu)} disabled={updating !== ''}>
              {updating === gpu ? '…' : $t('settings.gpuDownload', { gpu })}
            </button>
          </div>
          {#if updating === gpu && updateProgress}
            {@const pct = updateProgress.total
              ? Math.round((updateProgress.downloaded / updateProgress.total) * 100)
              : null}
            <div class="progress"><div class="bar" style={`width:${pct ?? 30}%`}></div></div>
            <p class="muted">
              {$t('settings.downloading')}: {fmtBytes(updateProgress.downloaded)}{#if updateProgress.total}
                / {fmtBytes(updateProgress.total)}{/if}{#if pct != null} · {pct}%{/if}
            </p>
          {/if}
        {:else}
          <p class="muted">{$t('settings.gpuNeedManifest', { gpu })}</p>
        {/if}
      {/if}
    {/if}
  </section>

  <section class="card">
    <h3>{$t('settings.updates')}</h3>
    <label class="field">
      <span>{$t('settings.manifestUrl')}</span>
      <input bind:value={settings.updateManifestUrl} placeholder="https://github.com/.../releases/latest/download/manifest.json" />
    </label>
    <label class="field">
      <span>{$t('settings.publicKey')}</span>
      <input bind:value={settings.updatePublicKey} placeholder="ed25519 public key (hex)" />
    </label>
    <div class="row">
      <button onclick={checkUpdate} disabled={checking || !hasTauri}
        >{checking ? $t('settings.checking') : $t('settings.checkUpdate')}</button
      >
    </div>
    <p class="muted">
      {$t('settings.current')}: <code>{backendUpdate.current || '—'}</code>
      {#if updateInfo} · {$t('settings.latest')}: <code>{updateInfo.remote}</code>{/if}
    </p>
    {#if backendUpdate.manifest_url}
      <p class="muted">
        {$t('settings.effectiveSource')}: <code>{backendUpdate.manifest_url}</code>
        {#if backendUpdate.builtin_url} · {$t('settings.builtinSource')}{/if}
      </p>
    {/if}
    {#if updateInfo}
      {#if updateInfo.update_available}
        <ul class="file-list">
          {#each updateInfo.backends as b}
            <li>
              <span class="file-name">{b}</span>
              <button onclick={() => applyUpdate(b)} disabled={updating !== ''}
                >{updating === b ? '…' : $t('settings.update')}</button
              >
            </li>
          {/each}
        </ul>
      {:else}
        <p class="ok">{$t('settings.upToDate')}</p>
      {/if}
    {/if}
    {#if updating && updateProgress}
      {@const pct = updateProgress.total
        ? Math.round((updateProgress.downloaded / updateProgress.total) * 100)
        : null}
      <div class="progress"><div class="bar" style={`width:${pct ?? 30}%`}></div></div>
      <p class="muted">
        {$t('settings.downloading')}: {fmtBytes(updateProgress.downloaded)}{#if updateProgress.total}
          / {fmtBytes(updateProgress.total)}{/if}{#if pct != null} · {pct}%{/if}
      </p>
    {/if}
    {#if updateMsg}<p class="ok">{$t('settings.updateDone')}: {updateMsg}</p>{/if}
  </section>

  {#if saved}<div class="ok">{$t('settings.saved')}</div>{/if}
  {#if error}<div class="error">{error}</div>{/if}
</div>
