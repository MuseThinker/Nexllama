<script>
import { onMount } from 'svelte';
import { invoke, hasTauri, pickFolder } from '../lib/tauri.js';
import { t } from '../lib/i18n.js';

const KEY = 'nexllama.settings';

const defaults = {
  mode: 'resident',
  keepAliveSecs: 0,
  backend: 'auto',
  autoStartDaemon: true,
  model: '',
  autoLoad: [],
  modelsDir: '',
  nGpuLayers: 999,
  nCtx: 4096,
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
  bindHost: '127.0.0.1',
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
let backendInfo = $state(null);
let modelsDirPath = $state('');
let localModels = $state([]);
let autoStart = $state(local().autoStartDaemon ?? true);
let busy = $state(false);
let backendUpdate = $state({ current: '', manifest_url: null, available: [] });
let updateInfo = $state(null);
let checking = $state(false);
let updating = $state('');
let updateMsg = $state('');
let copied = $state(false);

// Only loopback addresses are safe without a token; anything else is reachable
// from the network, so the UI has to say so out loud.
const exposed = $derived(
  settings.bindHost !== '127.0.0.1' && settings.bindHost !== 'localhost' && settings.bindHost !== '',
);

/// A fresh 24-byte token as hex.
///
/// `crypto.getRandomValues` is a browser API, so this works in the desktop shell
/// and in a plain browser alike — no reason to gate it on Tauri.
function generateKey() {
  const bytes = new Uint8Array(24);
  crypto.getRandomValues(bytes);
  settings.apiKey = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
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
}

async function refreshBackendUpdate() {
  if (!hasTauri) return;
  try {
    backendUpdate = await invoke('backend_update_status');
  } catch {
    /* ignore */
  }
}

async function checkUpdate() {
  checking = true;
  error = '';
  updateMsg = '';
  try {
    await invoke('set_settings', { settings });
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
  try {
    const r = await invoke('apply_backend_update', { backend });
    updateMsg = `${r.backend} → ${r.version}`;
    await refreshBackendUpdate();
    await refreshStatus();
  } catch (e) {
    error = String(e);
  } finally {
    updating = '';
  }
}

async function refreshBackend() {
  if (!hasTauri) return;
  try {
    backendInfo = await invoke('backend_info');
  } catch (e) {
    error = String(e);
  }
}

async function refreshModelsDir() {
  if (!hasTauri) return;
  try {
    modelsDirPath = await invoke('models_dir_cmd');
  } catch (e) {
    error = String(e);
  }
}

async function openFolder() {
  try {
    await invoke('open_models_dir');
  } catch (e) {
    error = String(e);
  }
}

async function refreshModels() {
  if (!hasTauri) return;
  try {
    localModels = await invoke('list_local_models');
  } catch {
    /* ignore */
  }
}

function toggleAutoLoad(path, checked) {
  const set = new Set(settings.autoLoad ?? []);
  if (checked) set.add(path);
  else set.delete(path);
  settings.autoLoad = [...set];
}

async function selectFolder() {
  const dir = await pickFolder($t('settings.selectFolder'));
  if (!dir) return;
  settings.modelsDir = dir;
  await persist();
  await refreshModelsDir();
  await refreshModels();
  await refresh();
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
  if (!Array.isArray(settings.autoLoad)) {
    settings.autoLoad = settings.model ? [settings.model] : [];
  }
  autoStart = settings.autoStartDaemon;
  refreshStatus();
  refreshBackend();
  refreshModelsDir();
  refreshModels();
  refreshBackendUpdate();
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
  const payload = { ...settings, autoStartDaemon: autoStart };
  localStorage.setItem(KEY, JSON.stringify(payload));
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
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => persist(), 600);
});

async function startDaemon() {
  busy = true;
  error = '';
  try {
    await invoke('daemon_start', {
      model: null,
      keep_alive_secs: settings.keepAliveSecs,
      n_gpu_layers: settings.nGpuLayers,
      n_ctx: settings.nCtx,
    });
    await refreshStatus();
  } catch (e) {
    error = String(e);
  } finally {
    busy = false;
  }
}

async function stopDaemon() {
  busy = true;
  try {
    await invoke('daemon_stop');
    await refreshStatus();
  } catch (e) {
    error = String(e);
  } finally {
    busy = false;
  }
}

function applyMode() {
  if (settings.mode === 'resident') settings.keepAliveSecs = 0;
  else if (settings.keepAliveSecs === 0) settings.keepAliveSecs = 300;
}

onMount(load);
</script>

<div class="page settings">
  <header>
    <h2>{$t('settings.title')}</h2>
    <button onclick={persist}>{$t('settings.save')}</button>
  </header>

  <section class="card">
    <h3>{$t('settings.daemon')}</h3>
    <p>
      {$t('settings.status')}:
      <strong class:on={daemonInfo.running}
        >{daemonInfo.running ? $t('settings.running') : $t('settings.stopped')}</strong
      >
      · {$t('settings.backend')} <strong>{daemonInfo.backend}</strong>
      · {$t('settings.port')} {daemonInfo.port}
    </p>
    {#if daemonInfo.binary}<p class="muted"><code>{daemonInfo.binary}</code></p>{/if}
    <div class="row">
      <button onclick={startDaemon} disabled={busy || !hasTauri}>{$t('settings.start')}</button>
      <button class="ghost" onclick={stopDaemon} disabled={busy || !hasTauri}>{$t('settings.stop')}</button>
      <button class="ghost" onclick={refreshStatus} disabled={!hasTauri}>{$t('models.refresh')}</button>
    </div>
    {#if !hasTauri}<p class="muted">{$t('settings.tDesktop')}</p>{/if}
  </section>

  <section class="card">
    <h3>{$t('settings.network')}</h3>
    <label class="field">
      <span>{$t('settings.bindHost')}</span>
      <select bind:value={settings.bindHost}>
        <option value="127.0.0.1">{$t('settings.bindLocal')}</option>
        <option value="0.0.0.0">{$t('settings.bindAll')}</option>
      </select>
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
    <p class="muted">{$t('settings.networkRestart')}</p>
  </section>

  <section class="card">
    <h3>{$t('settings.modelsDir')}</h3>
    <p class="muted"><code>{modelsDirPath || '—'}</code></p>
    <div class="row">
      <button onclick={selectFolder} disabled={!hasTauri}>{$t('settings.selectFolder')}</button>
      <button class="ghost" onclick={openFolder} disabled={!hasTauri}>{$t('settings.openFolder')}</button>
    </div>
  </section>

  <section class="card">
    <h3>{$t('settings.lifecycle')}</h3>
    <label class="field">
      <span>{$t('settings.mode')}</span>
      <select bind:value={settings.mode} onchange={applyMode}>
        <option value="on-demand">{$t('settings.modeOnDemand')}</option>
        <option value="resident">{$t('settings.modeResident')}</option>
      </select>
    </label>
    <label class="field">
      <span>{$t('settings.keepAlive')}</span>
      <input type="number" min="0" bind:value={settings.keepAliveSecs} />
    </label>
    <div class="field">
      <span>{$t('settings.autoLoadModel')}</span>
      {#if !hasTauri}
        <p class="muted">{$t('settings.tDesktop')}</p>
      {:else if localModels.length === 0}
        <p class="muted">{$t('models.libraryEmpty')}</p>
      {:else}
        <div class="checklist">
          {#each localModels as b}
            {#if b.main}
              <label class="checkline">
                <input
                  type="checkbox"
                  checked={(settings.autoLoad ?? []).includes(b.main)}
                  onchange={(e) => toggleAutoLoad(b.main, e.target.checked)}
                />
                <span>{b.name}</span>
              </label>
            {/if}
          {/each}
        </div>
      {/if}
    </div>
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
      <p class="muted">
        {$t('settings.preferred')}: {backendInfo.preferred} · {$t('settings.resolved')}: {backendInfo.resolved}
      </p>
      {#if backendInfo.gpus?.length}
        <p class="muted">{$t('settings.gpu')}: {backendInfo.gpus.join(', ')}</p>
      {/if}
      <p class="muted">{$t('settings.bundled')}: {backendInfo.available.join(', ') || '-'}</p>
    {/if}
  </section>

  <section class="card">
    <h3>{$t('settings.updates')}</h3>
    <label class="field">
      <span>{$t('settings.manifestUrl')}</span>
      <input bind:value={settings.updateManifestUrl} placeholder="https://github.com/.../releases/latest/download/manifest.json" />
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
    {#if updateMsg}<p class="ok">{$t('settings.updateDone')}: {updateMsg}</p>{/if}
  </section>

  <section class="card">
    <h3>{$t('settings.advanced')}</h3>
    <div class="grid">
      <label class="field">
        <span>{$t('models.gpuLayers')}</span>
        <input type="number" min="0" bind:value={settings.nGpuLayers} />
      </label>
      <label class="field">
        <span>{$t('settings.mainGpu')}</span>
        <input type="number" min="0" bind:value={settings.mainGpu} />
      </label>
      <label class="field">
        <span>{$t('models.context')}</span>
        <input type="number" min="0" step="512" bind:value={settings.nCtx} />
      </label>
      <label class="field">
        <span>{$t('settings.nBatch')}</span>
        <input type="number" min="1" bind:value={settings.nBatch} />
      </label>
      <label class="field">
        <span>{$t('settings.nUbatch')}</span>
        <input type="number" min="1" bind:value={settings.nUbatch} />
      </label>
      <label class="field">
        <span>{$t('settings.threads')}</span>
        <input type="number" min="0" bind:value={settings.threads} />
      </label>
      <label class="field">
        <span>{$t('settings.threadsBatch')}</span>
        <input type="number" min="0" bind:value={settings.threadsBatch} />
      </label>
      <label class="field">
        <span>{$t('settings.loadMode')}</span>
        <select bind:value={settings.loadMode}>
          <option value="auto">auto</option>
          <option value="mmap">mmap</option>
          <option value="none">none</option>
          <option value="mlock">mlock</option>
          <option value="mmap_mlock">mmap_mlock</option>
          <option value="dio">dio</option>
        </select>
      </label>
      <label class="field">
        <span>{$t('settings.flashAttn')}</span>
        <select bind:value={settings.flashAttn}>
          <option value="auto">auto</option>
          <option value="on">on</option>
          <option value="off">off</option>
        </select>
      </label>
      <label class="field">
        <span>{$t('settings.ropeFreqBase')}</span>
        <input type="number" min="0" step="1000" bind:value={settings.ropeFreqBase} />
      </label>
      <label class="field">
        <span>{$t('settings.ropeFreqScale')}</span>
        <input type="number" min="0" step="0.1" bind:value={settings.ropeFreqScale} />
      </label>
    </div>
    <div class="grid checks">
      <label class="checkline">
        <input type="checkbox" bind:checked={settings.useMlock} />
        <span>{$t('settings.useMlock')}</span>
      </label>
      <label class="checkline">
        <input type="checkbox" bind:checked={settings.offloadKqv} />
        <span>{$t('settings.offloadKqv')}</span>
      </label>
    </div>
  </section>

  {#if saved}<div class="ok">{$t('settings.saved')}</div>{/if}
  {#if error}<div class="error">{error}</div>{/if}
</div>
