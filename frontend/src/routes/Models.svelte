<script>
import { onMount } from 'svelte';
import { daemon } from '../lib/api.js';
import { t } from '../lib/i18n.js';
import { invoke, listen, hasTauri, pickFolder, openExternal } from '../lib/tauri.js';
import { CLOUD_CATALOG, catalogById } from '../lib/cloudCatalog.js';
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
  // 0 = auto: the daemon sizes the context to the machine (llama.cpp's own fit)
  // and caps it at the model's training length, so it is not too small to hold a
  // prompt nor large enough to exhaust VRAM.
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
  // Reuse the KV of a shared prefix across turns. On by default (verified by
  // prefix_check); ignored while a drafter is loaded.
  prefixCache: true,
  // Let a reasoning model think before answering. On by default; off makes the
  // reply start on the answer. A per-model setting, applied to every chat with
  // this model unless a request names its own.
  thinking: true,
  // MTP draft tokens per round (llama.cpp's --spec-draft-n-max). 0 leaves the
  // engine default (2). Higher drafts further ahead at a lower acceptance rate;
  // worth raising for models where the head is accurate.
  draftNMax: 0,
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

// Whether to load the bundled MTP head. The engine takes exactly one drafter,
// so choosing the side-mounted file means *not* loading the head — returning
// `true` here for the "draft" choice is what made a DFlash selection load MTP.
function wantMtp(b) {
  const choice = drafter[b.name] ?? 'auto';
  if (choice === 'mtp') return b.mtp === true;
  if (choice === 'draft') return false;
  if (choice === 'none') return false;
  // Auto: prefer the bundled head when the side file is another MTP head — the
  // bundled one shares the target's own weights instead of re-reading a copy.
  // A side DFlash/EAGLE checkpoint is the better drafter, so it wins there.
  if (b.mtp && (b.draft == null || b.draft_kind === 'MTP')) return true;
  return false;
}

const draftLabel = (b) => b.draft_kind ?? 'draft';

// What to call the drafter a loaded model actually uses, from the draft file's
// name. `info.mtp` is a property of the GGUF — "ships a head" — and says
// nothing about which drafter a reply came from.
const kindOf = (p) => {
  const n = String(p).toLowerCase();
  return n.includes('dflash')
    ? 'DFlash'
    : n.includes('eagle')
      ? 'EAGLE-3'
      : n.includes('mtp') || n.includes('nextn')
        ? 'MTP'
        : 'draft';
};

let query = $state('');
// Restricting a search to one author is the difference between hundreds of
// repos of uneven quality and a list worth reading. Empty means "anywhere".
let author = $state('');
let authors = $state([]);
let searching = $state(false);
let results = $state([]);
// A long search once lived on screen until the next search replaced it; being
// able to fold it away is how the rest of the page gets its room back.
let resultsOpen = $state(true);
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

// Everything the service knows about, resident or not. A model that was
// released to make room for another of its kind is still listed here: it is one
// step away from running again, and hiding it would make the list claim fewer
// models than the user loaded.
const loadedModels = $derived((status.models ?? []).filter((m) => !m.cloud));
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
  // Our own timeout, from `api.js`: whichever call it was, the sentence says how
  // long it waited and that the wait is what failed.
  if (e?.timedOut) return $t('models.requestTimeout', { secs: e.timeoutSecs });
  if (e instanceof TypeError && String(e).includes('fetch')) return $t('models.daemonDown');
  return String(e);
}

async function refresh() {
  try {
    status = await daemon.status();
    error = '';
  } catch (e) {
    // A poll that fails while a model is loading is a poll that timed out, not
    // a service that is down: the load itself has the port, and the panel would
    // otherwise report "not connected" over the top of a load that is about to
    // succeed. Only a quiet poll is worth telling the user about.
    if (!busy) error = describeError(e);
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
  // In a browser there is no folder picker and no local disk to read: the
  // library is whatever the daemon was pointed at with `--models-dir`, read with
  // the same `model-index` rules the app's own command uses. One code path for
  // "what is installed" is the point — two would drift.
  if (!hasTauri) {
    try {
      const listing = await daemon.localModels();
      localModels = listing?.models ?? [];
      modelsDir = listing?.dir ?? null;
      error = '';
    } catch (e) {
      localModels = [];
      error = describeError(e);
    }
    return;
  }
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
    thinking: adv.thinking !== false,
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
    if ((drafter[b.name] ?? 'auto') === 'none') {
      // Load the model alone: no head, no side file.
      body.load_mtp = false;
    } else if (wantMtp(b)) {
      body.load_mtp = true;
    } else {
      body.load_mtp = false;
      if (b.draft) body.draft_model = b.draft;
    }
    // Draft length, on whichever drafter this load uses: the bundled head reads
    // `mtp_n_max`, a side checkpoint reads `draft_n_max`. Sending `mtp_n_max` for
    // a side checkpoint would turn the bundled head on as well.
    const draftN = Number(adv.draftNMax);
    if (draftN > 0) {
      if (body.load_mtp) body.mtp_n_max = draftN;
      else if (body.draft_model) body.draft_n_max = draftN;
    }
    if (b.loras?.length) body.lora = b.loras;
    const res = await daemon.load(body);
    if (res?.notes?.length) notice = res.notes.join('; ');
    await refresh();
    await persistAutoLoad();
  } catch (e) {
    error = describeError(e);
  } finally {
    busy = '';
    loadingPath = '';
  }
}

/// Remember the local models that are loaded, so a restart (or an update) brings
/// the same ones back up. Clearing the last one clears the list.
async function persistAutoLoad() {
  if (!hasTauri) return;
  try {
    const s = await invoke('get_settings');
    const loaded = (status.models ?? [])
      .filter((m) => !m.cloud && m.loaded !== false && m.path)
      .map((m) => m.path);
    // Never erase the list because nothing happens to be resident: a poll taken
    // during a restart, or a model that has been released to make room, would
    // otherwise leave the next launch with nothing to load.
    if (loaded.length === 0) return;
    s.autoLoad = loaded;
    await invoke('set_settings', { settings: s });
  } catch {
    /* remembering is best-effort; a failure must not disturb the load */
  }
}

async function unload(id) {
  busy = id;
  try {
    await daemon.unload(id);
    await refresh();
    await persistAutoLoad();
  } catch (e) {
    error = describeError(e);
  } finally {
    busy = '';
  }
}

async function unloadAll() {
  busy = '*';
  try {
    await daemon.unload();
    await refresh();
    await persistAutoLoad();
  } catch (e) {
    error = describeError(e);
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
  resultsOpen = true;
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
  loadCloud();
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

// ── 云端模型 ─────────────────────────────────────────────────────────────
// Providers are non-secret config (id/name/baseUrl/models) kept in settings; the
// key/token for each lives in the OS credential store via the app, never here.
let view = $state('local');
let autoEnabled = $state(false);
let autoPrefer = $state('local');
let cloud = $state([]);
let cloudKeys = $state([]);
let cloudPicks = $state('');
let cloudBusy = $state('');
let cloudNotice = $state('');
let cloudError = $state('');

async function loadCloud() {
  if (!hasTauri) return;
  try {
    const s = await invoke('get_settings');
    autoEnabled = s?.synergyEnabled === true;
    autoPrefer = s?.synergyPrefer === 'cloud' ? 'cloud' : 'local';
    cloud = (Array.isArray(s?.cloudProviders) ? s.cloudProviders : []).map((p) => ({
      ...p,
      _key: '',
      _available: [],
      _pick: (p.models ?? [])[0] ?? '',
      _saved: false,
    }));
    const ids = cloud.map((p) => p.id);
    cloudKeys = ids.length ? await invoke('cloud_key_present', { ids }) : [];
  } catch (e) {
    cloudError = String(e);
  }
}

function addProvider() {
  const c = catalogById(cloudPicks);
  if (!c || cloud.some((p) => p.id === c.id)) return;
  cloud = [
    ...cloud,
    {
      id: c.id,
      name: c.name,
      baseUrl: c.baseUrl,
      console: c.console,
      auth: c.auth,
      models: [],
      enabled: true,
      _key: '',
      _available: [],
      _pick: '',
      _saved: false,
    },
  ];
  cloudPicks = '';
}

function removeProvider(id) {
  if (cloudKeys.includes(id)) invoke('cloud_key_clear', { id }).catch(() => {});
  cloudKeys = cloudKeys.filter((x) => x !== id);
  cloud = cloud.filter((p) => p.id !== id);
  saveCloud(true);
}

const hasKey = (id) => cloudKeys.includes(id);

/// Turn `auto` routing on/off.
async function setAutoEnabled(on) {
  autoEnabled = !!on;
  cloudError = '';
  try {
    await invoke('set_routing', { enabled: autoEnabled });
  } catch (e) {
    cloudError = String(e);
  }
}

/// Persist and push the `auto` alias policy (local-first or cloud-first).
async function setAutoPrefer(next) {
  autoPrefer = next === 'cloud' ? 'cloud' : 'local';
  cloudError = '';
  try {
    await invoke('set_routing', { prefer: autoPrefer });
  } catch (e) {
    cloudError = String(e);
  }
}

/// Candidate models for the picker: what the provider listed plus whatever is
/// already enabled, so a saved choice still shows after a reload.
function candidates(p) {
  return [...new Set([...(p._available ?? []), ...(p.models ?? [])])];
}

/// Store a provider's key the moment the field is left, so it persists even if
/// the user never presses save. Only ids cross back, never the value.
async function persistKey(p) {
  if (!p._key) return;
  try {
    await invoke('cloud_key_set', { id: p.id, key: p._key });
    if (!cloudKeys.includes(p.id)) cloudKeys = [...cloudKeys, p.id];
  } catch (e) {
    cloudError = String(e);
  }
}

async function fetchModels(p) {
  cloudBusy = p.id;
  cloudError = '';
  cloudNotice = '';
  try {
    const r = await invoke('cloud_fetch_models', {
      probe: { baseUrl: p.baseUrl, key: p._key || '' },
    });
    const list = Array.isArray(r?.models) ? r.models : [];
    p._available = list;
    if (!p._pick || !list.includes(p._pick)) p._pick = list[0] ?? p._pick;
    cloudNotice = $t('cloud.fetched', { n: list.length });
  } catch (e) {
    cloudError = String(e);
  } finally {
    cloudBusy = '';
  }
}

/// Confirm this provider: use the chosen model, then save and push. The key
/// travels with the push, so it is stored and forwarded in one step.
async function confirmProvider(p) {
  cloudBusy = p.id;
  cloudError = '';
  cloudNotice = '';
  try {
    if (p._pick) p.models = [p._pick];
    p._saved = true;
    await saveCloud(true);
    if (!cloudError) cloudNotice = $t('cloud.confirmed', { name: p.name });
  } catch (e) {
    p._saved = false;
    cloudError = String(e);
  } finally {
    cloudBusy = '';
  }
}

async function saveCloud(silent) {
  if (!hasTauri) {
    if (!silent) cloudNotice = $t('cloud.saved');
    return;
  }
  try {
    const s = await invoke('get_settings');
    // Drop the transient input fields; only real config is persisted.
    const clean = cloud.map(({ _key, _available, _pick, ...p }) => p);
    s.cloudProviders = clean;
    await invoke('set_settings', { settings: s });
    // The keys ride along here: the app stores them and forwards them to the
    // daemon, so a key only just typed is in effect immediately.
    const pushed = await invoke('cloud_push_providers', {
      providers: cloud.map((p) => ({
        id: p.id,
        name: p.name,
        baseUrl: p.baseUrl,
        models: p.models ?? [],
        key: p._key || '',
      })),
    });
    const keyed = cloud.filter((p) => p._key).map((p) => p.id);
    if (keyed.length) cloudKeys = [...new Set([...cloudKeys, ...keyed])];
    cloud = clean.map((p) => ({ ...p, _key: '', _available: [], _pick: (p.models ?? [])[0] ?? '' }));
    if (!silent) cloudNotice = $t('cloud.saved');
    // Saved, but nothing reached the daemon with a key: name it and drop the
    // "saved" tick so the button does not claim success that did not happen.
    if ((pushed?.count ?? 0) > 0 && (pushed?.withKey ?? 0) === 0) {
      cloudError = $t('cloud.noKey');
      cloud = cloud.map((p) => ({ ...p, _saved: false }));
    }
  } catch (e) {
    cloudError = String(e);
  }
}
</script>

<div class="page models">
  <header>
    <h2>{$t('models.title')}</h2>
    <div class="row">
      <button class="ghost" onclick={refreshAll}>{$t('models.refresh')}</button>
      {#if loadedModels.length > 0}
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
    {#if daemonInfo.gpu_backend && daemonInfo.backend !== daemonInfo.gpu_backend}
      <p class="muted">{$t('models.gpuNotUsed', { gpu: daemonInfo.gpu_backend })}</p>
    {/if}
    <h4 class="section-sub">{$t('models.available')}</h4>
    {#if (status.models ?? []).length === 0}
      <p class="muted">{$t('models.none')}</p>
    {:else}
      <ul class="model-list">
        {#each status.models as m (m.id)}
          <li>
            <span class="tag" class:cloud={m.cloud}>
              {m.cloud ? $t('models.badgeCloud') : $t('models.badgeLocal')}
            </span>
            <span class="file-name">{m.id}</span>
            {#if m.cloud}<span class="muted">{m.provider}</span>{/if}
            {#if !m.cloud && m.loaded === false}<span class="muted">· {$t('models.released')}</span>{/if}
          </li>
        {/each}
      </ul>
    {/if}
    <div class="synergy">
      <div class="synergy-head">
        <span class="synergy-title">
          {$t('models.synergy')}
          <span class="info" title={$t('models.autoHint')}>ⓘ</span>
        </span>
        <label class="toggle routing" title={$t('models.synergy')}>
          <input type="checkbox" checked={autoEnabled} onchange={(e) => setAutoEnabled(e.target.checked)} />
          <span class="track"></span>
        </label>
      </div>
      {#if autoEnabled}
        <label class="field synergy-policy">
          <span>
            {$t('models.autoPolicy')}
            <span
              class="info"
              title={autoPrefer === 'cloud' ? $t('models.autoCloudHint') : $t('models.autoLocalHint')}>ⓘ</span
            >
          </span>
          <select value={autoPrefer} onchange={(e) => setAutoPrefer(e.target.value)}>
            <option value="local">{$t('models.autoLocalFirst')}</option>
            <option value="cloud">{$t('models.autoCloudFirst')}</option>
          </select>
        </label>
      {/if}
    </div>
    {#if !hasTauri}<p class="muted">{$t('settings.tDesktop')}</p>{/if}
  </section>

  <div class="seg">
    <button class:active={view === 'local'} onclick={() => (view = 'local')}>{$t('models.localTab')}</button>
    <button class:active={view === 'cloud'} onclick={() => (view = 'cloud')}>{$t('models.cloudTab')}</button>
  </div>

  {#if view === 'local'}
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

    {#if results.length > 0 && resultsOpen}
      <div class="results-head">
        <span class="muted">{$t('models.results', { n: results.length })}</span>
        <button class="ghost" onclick={() => (resultsOpen = false)}>
          {$t('models.collapse')}
        </button>
      </div>
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
                  <option value="none">{$t('models.drafterNone')}</option>
                  {#if b.mtp}<option value="mtp">{$t('models.drafterMtpHead')}</option>{/if}
                  {#if b.draft}
                    <option value="draft">
                      {draftLabel(b)}{#if b.mtp} · {$t('models.drafterFile')}{/if}
                    </option>
                  {/if}
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
        <label class="field">
          <span>{$t('models.thinking')}</span>
          <select bind:value={adv.thinking}>
            <option value={true}>{$t('models.thinkingOn')}</option>
            <option value={false}>{$t('models.thinkingOff')}</option>
          </select>
        </label>
        <label class="field">
          <span>{$t('models.draftNMax')}</span>
          <input type="number" min="0" max="16" bind:value={adv.draftNMax} />
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
    <h3>{$t('models.loaded')} ({loadedModels.length})</h3>
    {#if loadedModels.length === 0}
      <p class="muted">{$t('models.none')}</p>
    {:else}
      <ul class="file-list">
        {#each loadedModels as m}
          <li>
            <span class="file-name">
              {m.id}
              {#if m.info?.vision}<span class="badge">vision</span>{/if}
              {#if m.draft_model}
                <span class="badge" title={m.draft_model}>{kindOf(m.draft_model)}</span>
              {:else if m.load_mtp && m.info?.mtp}
                <span class="badge">MTP</span>
              {/if}
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

  {:else}
  <!-- 云端模型：厂商配置 + 密钥 + 模型选择 -->
  <section class="card">
    <h3>{$t('models.cloud')}</h3>
    <p class="muted">{$t('models.cloudHint')}</p>
    <div class="row">
      <select bind:value={cloudPicks}>
        <option value="">{$t('models.cloudAdd')}</option>
        {#each CLOUD_CATALOG as c}
          <option value={c.id} disabled={cloud.some((p) => p.id === c.id)}>{c.name}</option>
        {/each}
      </select>
      <button onclick={addProvider} disabled={!cloudPicks}>{$t('models.cloudAddBtn')}</button>
    </div>
  </section>

  {#each cloud as p (p.id)}
    <section class="card">
      <div class="row space">
        <h3>{p.name}</h3>
        <button class="ghost" onclick={() => removeProvider(p.id)}>{$t('models.remove')}</button>
      </div>
      <div class="cloud-form">
        <label>{$t('models.cloudBaseUrl')}</label>
        <input bind:value={p.baseUrl} oninput={() => (p._saved = false)} />
        <button class="ghost" onclick={() => openExternal(p.console)}>{$t('models.getKey')} ↗</button>

        <label>
          {$t('models.cloudKey')}
          <span class="key-status" class:set={hasKey(p.id)}>
            {hasKey(p.id) ? $t('models.keySet') : $t('models.keyUnset')}
          </span>
        </label>
        <input
          type="password"
          bind:value={p._key}
          placeholder={hasKey(p.id) ? '••••••••' : ''}
          oninput={() => (p._saved = false)}
          onchange={() => persistKey(p)}
        />
        <button class="ghost" onclick={() => fetchModels(p)} disabled={cloudBusy === p.id}>
          {$t('models.fetchModels')}
        </button>

        <label>{$t('models.modelList')}</label>
        <select bind:value={p._pick} onchange={() => (p._saved = false)}>
          <option value="">{$t('models.modelPick')}</option>
          {#each candidates(p) as m}
            <option value={m}>{m}</option>
          {/each}
        </select>
        <button
          class="ghost"
          title={$t('models.confirm')}
          onclick={() => confirmProvider(p)}
          disabled={cloudBusy === p.id}
        >
          {p._saved ? $t('models.saved') : $t('models.confirm')}
        </button>
      </div>
    </section>
  {/each}

  {#if cloudNotice}<div class="notice">{cloudNotice}</div>{/if}
  {#if cloudError}<div class="error">{cloudError}</div>{/if}
  {/if}

  {#if notice}<div class="notice">{notice}</div>{/if}
  {#if error}<div class="error">{error}</div>{/if}
</div>
