<script>
import { onMount } from 'svelte';
import { daemon } from '../lib/api.js';
import { t } from '../lib/i18n.js';
import { invoke, listen, hasTauri, pickFolder } from '../lib/tauri.js';
import {
  startDaemon as daemonStart,
  stopDaemon as daemonStop,
  restartDaemon as daemonRestart,
} from '../lib/daemonctl.js';

// Load parameters. They are edited here because this is where they are used;
// on the desktop they are also written into the app settings, which is where
// the daemon picks them up from when it is spawned.
const advDefaults = {
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
  prefixCache: false,
};
const LOAD_KEY = 'nexllama.load';

let adv = $state({ ...advDefaults });
let status = $state({ models: [], default_model: null });
let error = $state('');
let notice = $state('');
let busy = $state('');
let loadingPath = $state('');
// Which drafter to load is a choice, because the engine takes exactly one: the
// bundled MTP head or a side-mounted draft file, never both. "auto" prefers the
// side-mounted file — a published DFlash/EAGLE checkpoint is the drafter
// someone made for this exact model — and falls back to the head.
let drafter = $state({});

function wantMtp(b) {
  const choice = drafter[b.name] ?? 'auto';
  if (choice === 'mtp') return b.mtp === true;
  if (choice === 'draft') return b.draft != null;
  return !b.draft && b.mtp === true;
}

const draftLabel = (b) => b.draft_kind ?? 'draft';

let query = $state('');
// Restricting a search to one author is the difference between hundreds of
// repos of uneven quality and a list worth reading. Empty means "anywhere".
let author = $state('');
let authors = $state([]);
let searching = $state(false);
let results = $state([]);
let activeRepo = $state('');
let files = $state([]);
let loadingFiles = $state(false);
let downloading = $state('');
let progress = $state({ downloaded: 0, total: 0 });
let lastDownloaded = $state('');
let localModels = $state([]);
// The inference service the models run on. It used to live on the Settings
// page, but starting it is the first thing this page needs, so it lives here.
let daemonInfo = $state({ running: false, port: 15656, backend: 'cpu' });
let daemonBusy = $state(false);

// Chat models first, then everything a user might keep for attachments later;
// the classes that cannot generate sit at the bottom where the Load button is
// clearly not for talking to.
const classRank = (b) => ({ vision: 1, llm: 2, speculative: 3, embedding: 4, rerank: 5 }[b.kind] ?? 3);
const orderedModels = $derived([...localModels].sort((a, b) => classRank(a) - classRank(b)));
let modelsDir = $state('');

let hydrated = false;
async function loadAdv() {
  let stored = null;
  if (hasTauri) {
    try {
      stored = await invoke('get_settings');
    } catch {
      /* ignore */
    }
  } else {
    try {
      stored = JSON.parse(localStorage.getItem(LOAD_KEY) ?? 'null');
    } catch {
      /* ignore */
    }
  }
  if (!stored) return;
  const picked = {};
  for (const k of Object.keys(advDefaults)) {
    if (stored[k] !== undefined && stored[k] !== null) picked[k] = stored[k];
  }
  // The effect below may flush before or after this assignment depending on the
  // platform path; dropping the flag here makes the run it causes count as the
  // load rather than as an edit, either way round.
  hydrated = false;
  adv = { ...advDefaults, ...picked };
}

let saveTimer;
$effect(() => {
  JSON.stringify(adv); // touch every field, so a keystroke anywhere retriggers
  if (!hydrated) {
    // The first run is the initial state, and the run right after a load is
    // the load itself — neither is a user edit.
    hydrated = true;
    return;
  }
  clearTimeout(saveTimer);
  saveTimer = setTimeout(persistAdv, 400);
});

function persistAdv() {
  if (hasTauri) {
    // The daemon is spawned with whatever is in this file, so the two have to
    // agree. Re-read first: the settings file holds keys this page does not own.
    invoke('get_settings')
      .then((s) => invoke('set_settings', { settings: { ...(s ?? {}), ...adv } }))
      .catch(() => {});
  } else {
    localStorage.setItem(LOAD_KEY, JSON.stringify(adv));
  }
}

async function refreshModelsDir() {
  if (!hasTauri) return;
  try {
    modelsDir = await invoke('models_dir_cmd');
  } catch {
    /* ignore */
  }
}

async function openFolder() {
  try {
    await invoke('open_models_dir');
  } catch (e) {
    error = String(e);
  }
}

async function selectFolder() {
  const dir = await pickFolder($t('settings.selectFolder'));
  if (!dir) return;
  try {
    const s = await invoke('get_settings');
    await invoke('set_settings', { settings: { ...s, modelsDir: dir } });
    await refreshModelsDir();
    await loadLocal();
  } catch (e) {
    error = String(e);
  }
}

// A daemon that is not listening fails every call the same way, and the raw
// browser text for it says nothing about what to do. Everything else is worth
// showing verbatim, because the daemon writes real sentences into it.
function describeError(e) {
  if (e instanceof TypeError && String(e).includes('fetch')) return $t('models.daemonDown');
  return String(e);
}

async function refresh() {
  try {
    status = await daemon.status();
    error = '';
  } catch (e) {
    error = describeError(e);
  }
}

async function refreshDaemon() {
  if (!hasTauri) return;
  try {
    daemonInfo = await invoke('daemon_status');
  } catch (e) {
    error = String(e);
  }
}

// Loading a model is a call to the daemon, so the service has to be up first.
// Starting it here is what makes Load work from a cold app; a start the app
// refuses — an exposed bind address with no access token — comes back as its
// own sentence rather than as a port nobody answers on.
async function ensureDaemon() {
  if (!hasTauri) return;
  await refreshDaemon();
  if (!daemonInfo.running) await daemonStart();
  await refreshDaemon();
}

// The header's Refresh covers the whole page: the loaded models and the service
// they run on. One button that refreshes everything beats two that differ by
// which card they sit in.
async function refreshAll() {
  await refresh();
  await refreshDaemon();
}

async function startDaemon() {
  daemonBusy = true;
  error = '';
  try {
    await daemonStart();
    await refreshDaemon();
    await refresh();
  } catch (e) {
    error = String(e);
  } finally {
    daemonBusy = false;
  }
}

async function stopDaemon() {
  daemonBusy = true;
  try {
    await daemonStop();
    await refreshDaemon();
    await refresh();
  } catch (e) {
    error = String(e);
  } finally {
    daemonBusy = false;
  }
}

async function restartDaemon() {
  daemonBusy = true;
  error = '';
  try {
    await daemonRestart();
    await refreshDaemon();
    await refresh();
  } catch (e) {
    error = String(e);
  } finally {
    daemonBusy = false;
  }
}

async function loadLocal() {
  if (!hasTauri) return;
  try {
    localModels = await invoke('list_local_models');
  } catch (e) {
    // An empty library and a read failure look identical from the outside.
    // Saying nothing turns "no models yet" into "where did everything go".
    error = String(e);
  }
}

function loadBody(path) {
  return {
    model: path,
    n_gpu_layers: Number(adv.nGpuLayers),
    n_ctx: Number(adv.nCtx),
    n_batch: Number(adv.nBatch),
    n_ubatch: Number(adv.nUbatch),
    main_gpu: Number(adv.mainGpu),
    threads: Number(adv.threads),
    threads_batch: Number(adv.threadsBatch),
    load_mode: adv.loadMode,
    use_mlock: adv.useMlock,
    flash_attn: adv.flashAttn,
    offload_kqv: adv.offloadKqv,
    rope_freq_base: Number(adv.ropeFreqBase),
    rope_freq_scale: Number(adv.ropeFreqScale),
    prefix_cache: adv.prefixCache === true,
  };
}

// Smart Fit. The daemon measures the checkpoint against what llama.cpp reports
// about this machine — the same helper llama-cli uses — so these are not guesses
// about layer sizes. Nothing is applied to the form unless the user asks.
let fitReport = $state(null);
let fitBusy = $state('');

async function checkFit(b) {
  if (!b?.main || busy || fitBusy) return;
  fitBusy = b.main;
  error = '';
  try {
    await ensureDaemon();
    const r = await daemon.fit({ model: b.main, n_ctx: Number(adv.nCtx) });
    fitReport = { path: b.main, ...r };
  } catch (e) {
    error = describeError(e);
  } finally {
    fitBusy = '';
  }
}

function adoptFit() {
  if (!fitReport) return;
  if (fitReport.suggested_gpu_layers != null) {
    adv.nGpuLayers = String(fitReport.suggested_gpu_layers);
  }
  if (fitReport.suggested_ctx != null) {
    adv.nCtx = String(fitReport.suggested_ctx);
  }
  notice = $t('models.fitAdopted');
}

async function loadBundle(b) {
  if (!b?.main || busy) return;
  busy = 'load';
  loadingPath = b.main;
  error = '';
  notice = '';
  try {
    await ensureDaemon();
    const body = loadBody(b.main);
    if (b.projector) body.mmproj = b.projector;
    if (wantMtp(b)) {
      body.load_mtp = true;
    } else {
      body.load_mtp = false;
      if (b.draft) body.draft_model = b.draft;
    }
    if (b.loras?.length) body.lora = b.loras;
    const res = await daemon.load(body);
    if (res?.notes?.length) notice = res.notes.join('; ');
    await refresh();
  } catch (e) {
    error = describeError(e);
  } finally {
    busy = '';
    loadingPath = '';
  }
}

async function unload(id) {
  busy = id;
  try {
    await daemon.unload(id);
    await refresh();
  } catch (e) {
    error = String(e);
  } finally {
    busy = '';
  }
}

async function unloadAll() {
  busy = '*';
  try {
    await daemon.unload();
    await refresh();
  } catch (e) {
    error = String(e);
  } finally {
    busy = '';
  }
}

async function search() {
  if (!hasTauri) {
    error = $t('models.tDesktop');
    return;
  }
  if (!query.trim()) return;
  searching = true;
  error = '';
  results = [];
  files = [];
  activeRepo = '';
  try {
    results = await invoke('hub_search', {
      query: query.trim(),
      author: author || null,
      limit: 20,
    });
  } catch (e) {
    error = String(e);
  } finally {
    searching = false;
  }
}

async function pickRepo(id) {
  // One listing at a time: two racing listings would pair one repo's files
  // with the other repo's name, and a download built from that pair asks for
  // a file that does not exist.
  if (loadingFiles) return;
  activeRepo = id;
  files = [];
  loadingFiles = true;
  error = '';
  try {
    const got = await invoke('hub_list_gguf', { repo: id });
    // Another repo may have been picked while this one was in flight; a late
    // answer belongs to a request nobody is waiting for any more.
    if (activeRepo !== id) return;
    files = got;
  } catch (e) {
    if (activeRepo === id) error = String(e);
  } finally {
    if (activeRepo === id) loadingFiles = false;
  }
}

async function download(file) {
  if (!hasTauri) {
    error = $t('models.tDesktop');
    return;
  }
  downloading = file.filename;
  progress = { downloaded: 0, total: file.size };
  error = '';
  notice = '';
  try {
    // Checked *before* the transfer, not after: warning once several gigabytes
    // have already landed is not a warning, it is a receipt.
    if (file.size > 0) {
      try {
        await ensureDaemon();
        const r = await daemon.fit({ size_bytes: Number(file.size) });
        if (!r.fits) notice = r.reason;
      } catch {
        // A failed check must never block the download. It is advice, and
        // whoever is clicking may know this machine better than the probe does.
      }
    }
    const path = await invoke('hub_download', { repo: activeRepo, filename: file.filename });
    lastDownloaded = path;
    await loadLocal();
  } catch (e) {
    error = String(e);
  } finally {
    downloading = '';
    progress = { downloaded: 0, total: 0 };
  }
}

function fmt(bytes) {
  if (!bytes) return '?';
  return (bytes / 1e6).toFixed(1) + ' MB';
}

function pct() {
  if (!progress.total) return 0;
  return Math.min(100, (progress.downloaded / progress.total) * 100);
}

onMount(() => {
  let un1;
  let un2;
  listen('download://progress', (e) => {
    // Progress events carry the file they belong to; adopting whichever event
    // arrived last would show one download's numbers on another's bar.
    if (e.payload.filename !== downloading) return;
    progress = { downloaded: e.payload.downloaded, total: e.payload.total ?? progress.total };
  }).then((f) => (un1 = f));
  listen('download://done', (e) => {
    lastDownloaded = e.payload.path;
    loadLocal();
  }).then((f) => (un2 = f));
  refresh();
  loadLocal();
  loadAdv();
  refreshModelsDir();
  refreshDaemon();
  // The list of authors lives next to the search that uses it, so adding one
  // does not mean editing the UI twice.
  if (hasTauri) {
    invoke('hub_trusted_authors')
      .then((list) => (authors = list ?? []))
      .catch(() => (authors = []));
  }
  return () => {
    un1?.();
    un2?.();
  };
});
</script>

<div class="page models">
  <header>
    <h2>{$t('models.title')}</h2>
    <div class="row">
      <button class="ghost" onclick={refreshAll}>{$t('models.refresh')}</button>
      {#if status.models.length > 0}
        <button class="ghost" onclick={unloadAll} disabled={busy !== ''}>{$t('models.unload')}</button>
      {/if}
    </div>
  </header>

  <!-- 推理服务：模型跑在它上面，先有服务才有下面的一切 -->
  <section class="card">
    <h3>{$t('settings.daemon')}</h3>
    <div class="daemon-status">
      <span class="daemon-dot" class:on={daemonInfo.running}></span>
      <div class="daemon-status-text">
        <strong class:on={daemonInfo.running}
          >{daemonInfo.running ? $t('settings.running') : $t('settings.stopped')}</strong
        >
        <span>{daemonInfo.running
            ? $t('settings.daemonRunningHint')
            : $t('settings.daemonStoppedHint')}</span>
      </div>
      <div class="daemon-actions">
        <button onclick={startDaemon} disabled={daemonBusy || !hasTauri || daemonInfo.running}>
          {$t('settings.start')}
        </button>
        <button class="ghost" onclick={stopDaemon} disabled={daemonBusy || !hasTauri || !daemonInfo.running}>
          {$t('settings.stop')}
        </button>
        <button
          class="ghost"
          onclick={restartDaemon}
          disabled={daemonBusy || !hasTauri || !daemonInfo.running}
          title={$t('settings.restartHint')}
        >
          {$t('settings.restart')}
        </button>
      </div>
    </div>
    {#if !daemonInfo.running && daemonInfo.last_error}
      <div class="error">{daemonInfo.last_error}</div>
    {/if}
    <dl class="daemon-facts">
      <div>
        <dt>{$t('settings.backend')}</dt>
        <dd>{daemonInfo.backend}</dd>
      </div>
      <div>
        <dt>{$t('settings.port')}</dt>
        <dd>{daemonInfo.port}</dd>
      </div>
    </dl>
    {#if !hasTauri}<p class="muted">{$t('settings.tDesktop')}</p>{/if}
  </section>

  <!-- 模型位置 -->
  <section class="card">
    <h3>{$t('models.dir')}</h3>
    <p class="muted"><code>{modelsDir || '—'}</code></p>
    <div class="row">
      <button onclick={selectFolder} disabled={!hasTauri}>{$t('settings.selectFolder')}</button>
      <button class="ghost" onclick={openFolder} disabled={!hasTauri}>{$t('settings.openFolder')}</button>
    </div>
  </section>

  <!-- 搜索并下载 -->
  <section class="card">
    <h3>{$t('models.download')}</h3>
    <div class="searchbar">
      <input
        bind:value={query}
        placeholder={$t('models.searchPlaceholder')}
        onkeydown={(e) => e.key === 'Enter' && search()}
      />
      <button onclick={search} disabled={searching}
        >{searching ? $t('models.searching') : $t('models.search')}</button
      >
    </div>

    {#if authors.length > 0}
      <div class="sources">
        <span class="sources-label">{$t('models.sources')}</span>
        <button class="chip" class:active={author === ''} onclick={() => (author = '')}>
          {$t('models.anyAuthor')}
        </button>
        {#each authors as a}
          <button class="chip" class:active={author === a} onclick={() => (author = a)}>{a}</button>
        {/each}
      </div>
    {/if}

    {#if !hasTauri}
      <p class="muted">{$t('models.tDesktop')}</p>
    {/if}

    {#if results.length > 0}
      <ul class="repo-list">
        {#each results as r}
          <li>
            <button class="repo" class:active={activeRepo === r.id} onclick={() => pickRepo(r.id)}>
              <span class="repo-id">{r.id}</span>
              <span class="repo-meta">{r.downloads.toLocaleString()} dl</span>
            </button>
          </li>
        {/each}
      </ul>
    {/if}

    {#if files.length > 0}
      <ul class="file-list">
        {#each files as f}
          <li>
            <span class="file-name">{f.filename}</span>
            <span class="file-size">{fmt(f.size)}</span>
            <button onclick={() => download(f)} disabled={downloading !== ''}>
              {downloading === f.filename ? $t('models.downloading') : $t('models.downloadBtn')}
            </button>
          </li>
        {/each}
      </ul>
    {/if}

    {#if downloading}
      <div class="progress"><div class="bar" style="width:{pct()}%"></div></div>
      <p class="muted">{downloading} — {pct().toFixed(1)}% ({fmt(progress.downloaded)})</p>
    {/if}

    {#if lastDownloaded}
      <p class="ok">{$t('models.saved')}: <code>{lastDownloaded}</code></p>
    {/if}
  </section>

  <!-- 本地模型库 -->
  <section class="card">
    <h3>{$t('models.library')}</h3>
    <p class="muted">{$t('models.libraryHint')}</p>
    {#if !hasTauri}
      <p class="muted">{$t('models.tDesktop')}</p>
    {:else if localModels.length === 0}
      <p class="muted">{$t('models.libraryEmpty')}</p>
    {:else}
      <ul class="file-list">
        {#each orderedModels as b}
          <li>
            <span class="file-name">
              {b.name}
              {#if b.kind === 'vision'}<span class="badge">vision</span>{/if}
              {#if b.kind === 'speculative'}<span class="badge" title={b.draft}>spec</span>{/if}
              {#if b.mtp}<span class="badge">MTP</span>{/if}
              {#if b.class === 'embedding' || b.class === 'rerank'}
                <span class="badge api">API · {b.class}</span>
              {/if}
            </span>
            <span class="file-size">
              {fmt(b.size)}{#if b.files.length > 1} · {b.files.length} {$t('models.files')}{/if}
            </span>
            {#if b.mtp || b.draft}
              <label class="checkline" title={$t('models.drafterHint')}>
                <select
                  value={drafter[b.name] ?? 'auto'}
                  onchange={(e) => (drafter[b.name] = e.target.value)}
                >
                  <option value="auto">{$t('models.drafterAuto')}</option>
                  {#if b.mtp}<option value="mtp">MTP</option>{/if}
                  {#if b.draft}<option value="draft">{draftLabel(b)}</option>{/if}
                </select>
              </label>
            {/if}
            <button class="ghost" onclick={() => checkFit(b)} disabled={busy !== '' || fitBusy !== ''}>
              {fitBusy === b.main ? $t('models.fitting') : $t('models.fit')}
            </button>
            <button onclick={() => loadBundle(b)} disabled={busy !== ''}>
              {loadingPath === b.main ? $t('models.loading') : $t('models.load')}
            </button>
            {#if fitReport?.path === b.main}
              <div class="fit-report" class:warn={!fitReport.fits}>
                <span>{fitReport.reason}</span>
                {#if fitReport.needed_mb > 0}
                  <span class="file-size">{fitReport.needed_mb} / {fitReport.available_mb} MiB</span>
                {/if}
                {#if fitReport.suggested_gpu_layers != null || fitReport.suggested_ctx != null}
                  <button class="ghost" onclick={adoptFit}>{$t('models.fitAdopt')}</button>
                {/if}
              </div>
            {/if}
          </li>
        {/each}
      </ul>
    {/if}

    <details class="advanced">
      <summary>{$t('models.advanced')}</summary>
      <div class="grid">
        <label class="field">
          <span>{$t('models.gpuLayers')}</span>
          <input type="number" min="0" bind:value={adv.nGpuLayers} />
        </label>
        <label class="field">
          <span>{$t('settings.mainGpu')}</span>
          <input type="number" min="0" bind:value={adv.mainGpu} />
        </label>
        <label class="field">
          <span>{$t('models.context')}</span>
          <input type="number" min="0" step="512" bind:value={adv.nCtx} />
        </label>
        <label class="field">
          <span>{$t('settings.nBatch')}</span>
          <input type="number" min="1" bind:value={adv.nBatch} />
        </label>
        <label class="field">
          <span>{$t('settings.nUbatch')}</span>
          <input type="number" min="1" bind:value={adv.nUbatch} />
        </label>
        <label class="field">
          <span>{$t('settings.threads')}</span>
          <input type="number" min="0" bind:value={adv.threads} />
        </label>
        <label class="field">
          <span>{$t('settings.threadsBatch')}</span>
          <input type="number" min="0" bind:value={adv.threadsBatch} />
        </label>
        <label class="field">
          <span>{$t('settings.loadMode')}</span>
          <select bind:value={adv.loadMode}>
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
          <select bind:value={adv.flashAttn}>
            <option value="auto">auto</option>
            <option value="on">on</option>
            <option value="off">off</option>
          </select>
        </label>
        <label class="field">
          <span>{$t('settings.ropeFreqBase')}</span>
          <input type="number" min="0" step="1000" bind:value={adv.ropeFreqBase} />
        </label>
        <label class="field">
          <span>{$t('settings.ropeFreqScale')}</span>
          <input type="number" min="0" step="0.1" bind:value={adv.ropeFreqScale} />
        </label>
      </div>
      <div class="grid checks">
        <label class="checkline">
          <input type="checkbox" bind:checked={adv.useMlock} />
          <span>{$t('settings.useMlock')}</span>
        </label>
        <label class="checkline">
          <input type="checkbox" bind:checked={adv.offloadKqv} />
          <span>{$t('settings.offloadKqv')}</span>
        </label>
        <label class="checkline" title={$t('settings.prefixCacheHint')}>
          <input type="checkbox" bind:checked={adv.prefixCache} />
          <span>{$t('settings.prefixCache')}</span>
        </label>
      </div>
    </details>
  </section>

  <!-- 已加载（运行中） -->
  <section class="card">
    <h3>{$t('models.loaded')} ({status.models.length})</h3>
    {#if status.models.length === 0}
      <p class="muted">{$t('models.none')}</p>
    {:else}
      <ul class="file-list">
        {#each status.models as m}
          <li>
            <span class="file-name">
              {m.id}
              {#if m.info?.vision}<span class="badge">vision</span>{/if}
              {#if m.info?.mtp}<span class="badge">MTP {m.load_mtp ? 'on' : 'off'}</span>{/if}
              {#if m.draft_model}<span class="badge" title={m.draft_model}>draft</span>{/if}
              {#if m.mmproj}<span class="badge" title={m.mmproj}>mmproj</span>{/if}
              {#if m.info?.model_class === 'embedding' || m.info?.model_class === 'rerank'}
                <span class="badge api">API · {m.info.model_class}</span>
              {/if}
              {#if m.id === status.default_model}<span class="badge">{$t('models.default')}</span>{/if}
            </span>
            <span class="file-size">ctx {m.n_ctx} · GPU {m.n_gpu_layers} · {$t('models.idle')} {m.idle_secs}s</span>
            <button class="ghost" onclick={() => unload(m.id)} disabled={busy !== ''}>{$t('models.unload')}</button>
          </li>
        {/each}
      </ul>
    {/if}
  </section>

  {#if notice}<div class="notice">{notice}</div>{/if}
  {#if error}<div class="error">{error}</div>{/if}
</div>
