<script>
import { onMount } from 'svelte';
import { daemon } from '../lib/api.js';
import { t } from '../lib/i18n.js';
import { invoke, listen, hasTauri, pickFolder } from '../lib/tauri.js';

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
};

let modelPath = $state('');
let adv = $state({ ...advDefaults });
let status = $state({ models: [], default_model: null });
let error = $state('');
let notice = $state('');
let busy = $state('');
let loadingPath = $state('');

let query = $state('');
let searching = $state(false);
let results = $state([]);
let activeRepo = $state('');
let files = $state([]);
let loadingFiles = $state(false);
let downloading = $state('');
let progress = $state({ downloaded: 0, total: 0 });
let lastDownloaded = $state('');
let localModels = $state([]);

// Chat models first, then everything a user might keep for attachments later;
// the classes that cannot generate sit at the bottom where the Load button is
// clearly not for talking to.
const classRank = (b) => ({ vision: 1, llm: 2, speculative: 3, embedding: 4, rerank: 5 }[b.kind] ?? 3);
const orderedModels = $derived([...localModels].sort((a, b) => classRank(a) - classRank(b)));
let modelsDir = $state('');

async function loadDefaults() {
  let stored = null;
  if (hasTauri) {
    try {
      stored = await invoke('get_settings');
    } catch {
      /* ignore */
    }
  } else {
    try {
      stored = JSON.parse(localStorage.getItem('nexllama.settings') ?? 'null');
    } catch {
      /* ignore */
    }
  }
  if (stored) adv = { ...advDefaults, ...stored };
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

async function refresh() {
  try {
    status = await daemon.status();
    error = '';
  } catch (e) {
    error = String(e);
  }
}

async function loadLocal() {
  if (!hasTauri) return;
  try {
    localModels = await invoke('list_local_models');
  } catch {
    /* ignore */
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
  };
}

async function load(path) {
  const p = (path ?? modelPath).trim();
  if (!p || busy) return;
  busy = 'load';
  loadingPath = p;
  error = '';
  notice = '';
  try {
    const res = await daemon.load(loadBody(p));
    if (res?.notes?.length) notice = res.notes.join('; ');
    await refresh();
  } catch (e) {
    error = String(e);
  } finally {
    busy = '';
    loadingPath = '';
  }
}

async function loadBundle(b) {
  if (!b?.main || busy) return;
  busy = 'load';
  loadingPath = b.main;
  error = '';
  notice = '';
  try {
    const body = loadBody(b.main);
    if (b.projector) body.mmproj = b.projector;
    if (b.draft) body.draft_model = b.draft;
    if (b.loras?.length) body.lora = b.loras;
    const res = await daemon.load(body);
    if (res?.notes?.length) notice = res.notes.join('; ');
    await refresh();
  } catch (e) {
    error = String(e);
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
    results = await invoke('hub_search', { query: query.trim(), limit: 20 });
  } catch (e) {
    error = String(e);
  } finally {
    searching = false;
  }
}

async function pickRepo(id) {
  activeRepo = id;
  files = [];
  loadingFiles = true;
  error = '';
  try {
    files = await invoke('hub_list_gguf', { repo: id });
  } catch (e) {
    error = String(e);
  } finally {
    loadingFiles = false;
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
  try {
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
    progress = { downloaded: e.payload.downloaded, total: e.payload.total ?? progress.total };
  }).then((f) => (un1 = f));
  listen('download://done', (e) => {
    lastDownloaded = e.payload.path;
    loadLocal();
  }).then((f) => (un2 = f));
  refresh();
  loadLocal();
  loadDefaults();
  refreshModelsDir();
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
      <button class="ghost" onclick={refresh}>{$t('models.refresh')}</button>
      {#if status.models.length > 0}
        <button class="ghost" onclick={unloadAll} disabled={busy !== ''}>{$t('models.unload')}</button>
      {/if}
    </div>
  </header>

  <!-- 1. 模型位置 -->
  <section class="card">
    <h3>1 · {$t('models.dir')}</h3>
    <p class="muted"><code>{modelsDir || '—'}</code></p>
    <div class="row">
      <button onclick={selectFolder} disabled={!hasTauri}>{$t('settings.selectFolder')}</button>
      <button class="ghost" onclick={openFolder} disabled={!hasTauri}>{$t('settings.openFolder')}</button>
    </div>
  </section>

  <!-- 2. 搜索并下载 -->
  <section class="card">
    <h3>2 · {$t('models.download')}</h3>
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

  <!-- 3. 本地模型库 -->
  <section class="card">
    <h3>3 · {$t('models.library')}</h3>
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
              {#if b.kind === 'speculative'}<span class="badge">spec</span>{/if}
              {#if b.class === 'embedding' || b.class === 'rerank'}
                <span class="badge api">API · {b.class}</span>
              {/if}
            </span>
            <span class="file-size">
              {fmt(b.size)}{#if b.files.length > 1} · {b.files.length} {$t('models.files')}{/if}
            </span>
            <button onclick={() => loadBundle(b)} disabled={busy !== ''}>
              {loadingPath === b.main ? $t('models.loading') : $t('models.load')}
            </button>
          </li>
        {/each}
      </ul>
    {/if}
  </section>

  <!-- 4. 已加载（运行中） -->
  <section class="card">
    <h3>4 · {$t('models.loaded')} ({status.models.length})</h3>
    {#if status.models.length === 0}
      <p class="muted">{$t('models.none')}</p>
    {:else}
      <ul class="file-list">
        {#each status.models as m}
          <li>
            <span class="file-name">
              {m.id}
              {#if m.info?.vision}<span class="badge">vision</span>{/if}
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

  <!-- 5. 从磁盘加载（高级） -->
  <section class="card">
    <h3>5 · {$t('models.loadDisk')}</h3>
    <label class="field">
      <span>{$t('models.path')}</span>
      <input bind:value={modelPath} placeholder="D:\\models\\qwen2.5-0.5b-instruct-q4_k_m.gguf" />
    </label>
    <button onclick={() => load()} disabled={busy !== ''}>
      {busy === 'load' ? $t('models.loading') : $t('models.loadModel')}
    </button>

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
      </div>
    </details>
  </section>

  {#if notice}<div class="notice">{notice}</div>{/if}
  {#if error}<div class="error">{error}</div>{/if}
</div>
