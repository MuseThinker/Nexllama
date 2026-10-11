<script>
import { onDestroy, onMount } from 'svelte';
import { daemon } from '../lib/api.js';
import { t } from '../lib/i18n.js';

let messages = $state([]);
let input = $state('');
let busy = $state(false);
let error = $state('');
let temperature = $state(0.8);
let topK = $state(40);
let topP = $state(0.95);
let minP = $state(0.05);
let repeatPenalty = $state(1.0);
let repeatLastN = $state(64);
let maxTokens = $state(0);
let seedMode = $state('random');
let seed = $state(0);
let speculative = $state(false);
let ngramN = $state(4);
let ngramM = $state(8);
let models = $state([]);
let apiOnly = $state(0);
let selected = $state('');
// The drafter belongs to the model that was loaded with one: a bundled MTP
// head or a side-mounted checkpoint drafts for it, and the n-gram drafter is
// the only thing the two numbers below still describe.
const active = $derived(models.find((m) => m.id === selected) ?? null);
const modelDrafts = $derived(!!(active?.draft_model || active?.load_mtp));
const drafter = $derived(
  active?.draft_model ? String(active.draft_model).split(/[\\/]/).pop() : active?.load_mtp ? 'MTP' : ''
);
let images = $state([]);
let fileInput;
// Set while a reply is streaming so it can be stopped. Aborting the request is
// what lets the daemon stop generating instead of finishing the reply into a
// socket that is no longer being read.
let controller = null;

// The transcript follows the reply while it streams — but only while the
// reader is already near the bottom. Someone scrolled up to reread a turn must
// not be dragged down on every token; that is worse than not following at all.
// "Near" rather than "exactly", because sub-pixel rounding can leave the
// scroll position a fraction off the true end and kill the pin forever.
let transcript;
let pinned = $state(true);

function onTranscriptScroll() {
  const el = transcript;
  if (!el) return;
  pinned = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
}

// Runs whenever the tail of the conversation changes — a new message, or the
// growing text of the reply now streaming. Reading the two values is what
// subscribes the effect to them.
$effect(() => {
  void messages.length;
  void messages[messages.length - 1]?.content;
  if (!pinned || !transcript) return;
  // Instant, not smooth: this fires per token, and easing between them reads
  // as the transcript lagging behind the text.
  transcript.scrollTo({ top: transcript.scrollHeight });
});

// The sampling panel. A native <details> only closes on its own summary, so
// once it was open it stayed open through every click elsewhere on the page.
let paramsOpen = $state(false);
let paramsEl = null;

function closeParamsIfOutside(e) {
  if (paramsEl && !paramsEl.contains(e.target)) paramsOpen = false;
}

// Largest attachment we will inline. Vision models re-read the image on every
// turn of a conversation, so the bytes are sent again with each request.
const MAX_IMAGE_BYTES = 4 * 1024 * 1024;

async function addImages(files) {
  error = '';
  for (const file of files) {
    if (!file.type.startsWith('image/')) continue;
    if (file.size > MAX_IMAGE_BYTES) {
      error = `${file.name}: ${(file.size / 1048576).toFixed(1)} MB exceeds the 4 MB limit`;
      continue;
    }
    let b64;
    try {
      b64 = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result).split(',')[1]);
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(file);
      });
    } catch {
      // One unreadable file must not abandon the rest of the selection.
      error = `${file.name}: could not be read`;
      continue;
    }
    images = [...images, { name: file.name, type: file.type, b64, url: URL.createObjectURL(file) }];
  }
  if (fileInput) fileInput.value = '';
}

/// An object URL pins the whole image in memory until it is revoked. Sent
/// attachments used to be pinned for the life of the tab, so a long session
/// with pictures held every one of them twice — once as the blob, once as the
/// base64 in the request.
function releaseUrls(list) {
  for (const m of list ?? []) {
    for (const img of m.images ?? []) {
      if (img.url) URL.revokeObjectURL(img.url);
    }
  }
}

onDestroy(() => releaseUrls(messages));

function removeImage(i) {
  const [dropped] = images.splice(i, 1);
  images = [...images];
  if (dropped?.url) URL.revokeObjectURL(dropped.url);
}

const CHAT_MODEL_KEY = 'nexllama.chatModel';

let refreshing = false;

async function refreshModels() {
  // Skip a poll whose predecessor is still waiting: the daemon can be busy
  // loading for a long time, and stacked requests only crowd out the ones that
  // matter.
  if (refreshing) return;
  refreshing = true;
  try {
    const s = await daemon.status();
    // Only generation models belong here. Embedding and rerank models serve the
    // HTTP API; they decode probabilities, not text.
    const chatty = (s.models ?? []).filter(
      (m) => !['embedding', 'rerank'].includes(m.info?.model_class),
    );
    apiOnly = (s.models ?? []).length - chatty.length;
    models = chatty;
    // Default order: the model the user last chose, else the daemon's
    // first-loaded one, else whatever is left. The daemon's default may well be
    // an embedding model, so it is only a preference, never an answer.
    if (!chatty.some((m) => m.id === selected)) {
      const remembered = localStorage.getItem(CHAT_MODEL_KEY);
      const pick =
        chatty.find((m) => m.id === remembered) ??
        chatty.find((m) => m.id === s.default_model) ??
        chatty[0];
      selected = pick?.id ?? '';
    }
  } catch {
    /* daemon offline */
  } finally {
    refreshing = false;
  }
}

function rememberModel(id) {
  selected = id;
  try {
    localStorage.setItem(CHAT_MODEL_KEY, id);
  } catch {
    /* private mode */
  }
}

// The daemon reports a failure either as a sentence or as a JSON error object.
// Interpolating the object straight into the template printed `[object Object]`
// — the one message that says nothing about what went wrong.
function errorText(ev) {
  const e = ev?.error;
  if (typeof e === 'string' && e.trim()) return e;
  if (e && typeof e === 'object') return String(e.message ?? JSON.stringify(e));
  if (typeof ev?.message === 'string' && ev.message.trim()) return ev.message;
  return 'generation failed';
}

async function send() {
  const text = input.trim();
  if ((!text && images.length === 0) || busy) return;

  error = '';
  const pending = [...images];
  messages = [...messages, { role: 'user', content: text, images: pending }];
  input = '';
  images = [];
  busy = true;
  controller = new AbortController();

  // The daemon speaks OpenAI-style content parts, so attachments are sent as
  // data URLs rather than as a second request.
  const wire = messages.map((m) => ({
    role: m.role,
    content:
      m.images?.length > 0
        ? [
            ...(m.content ? [{ type: 'text', text: m.content }] : []),
            ...m.images.map((data) => ({
              type: 'image_url',
              image_url: { url: `data:${data.type};base64,${data.b64}` },
            })),
          ]
        : m.content,
  }));

  const history = wire;
  messages = [...messages, { role: 'assistant', content: '' }];

  try {
    for await (const ev of daemon.stream(history, {
      signal: controller.signal,
      model: selected || undefined,
      temperature,
      top_k: topK,
      top_p: topP,
      min_p: minP,
      repeat_penalty: repeatPenalty,
      repeat_last_n: repeatLastN,
      max_tokens: maxTokens,
      speculative,
      ...(speculative ? { ngram_n: Number(ngramN), ngram_m: Number(ngramM) } : {}),
      ...(seedMode === 'fixed' ? { seed: Number(seed) } : {}),
    })) {
      if (ev.event === 'error' || ev.error) {
        error = errorText(ev);
        break;
      }
      if (ev.stats) {
        messages[messages.length - 1].stats = ev.stats;
        continue;
      }
      const delta = ev?.choices?.[0]?.delta?.content;
      if (delta) {
        // `$state` proxies deeply, so writing through the object is enough.
        // Copying the whole array per token made a long reply cost O(n²) and
        // re-diffed the entire transcript on every token.
        messages[messages.length - 1].content += delta;
      }
    }
  } catch (e) {
    // Stopping is not a failure — the reply is simply shorter than it would
    // have been. Anything else is worth showing.
    if (e?.name !== 'AbortError') error = String(e);
  } finally {
    busy = false;
    controller = null;
  }
}

function stop() {
  controller?.abort();
}

function onKeydown(e) {
  // Enter confirms a candidate in Chinese and Japanese IMEs. Sending on it
  // posts a half-composed sentence, so the composing state wins.
  if (e.isComposing) return;
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send();
  }
}

function tps(stats) {
  if (!stats || !stats.gen_ms) return '0.0';
  return (stats.generated_tokens / (stats.gen_ms / 1000)).toFixed(1);
}

// Qwen-style reasoning models wrap their chain of thought in ` thinking…`.
// Shown raw those tags are noise in the reply, so the text is split into the
// reasoning and the answer. `open` is true until the closing tag arrives, which
// is how a live stream is told from a finished reply.
const THINK_OPEN = ' thinking';
const THINK_CLOSE = '<' + '/think' + '>';

function splitThinking(text) {
  const start = text.indexOf(THINK_OPEN);
  if (start === -1) return { thinking: '', answer: text, has: false, open: false };
  const from = start + THINK_OPEN.length;
  const end = text.indexOf(THINK_CLOSE, from);
  if (end === -1) {
    return { thinking: text.slice(from), answer: '', has: true, open: true };
  }
  return {
    thinking: text.slice(from, end),
    answer: text.slice(end + THINK_CLOSE.length).replace(/^\n+/, ''),
    has: true,
    open: false,
  };
}

onMount(() => {
  refreshModels();
  const id = setInterval(refreshModels, 5000);
  // `pointerdown` rather than `click` so the panel is gone by the time the
  // click lands, instead of one frame after it.
  document.addEventListener('pointerdown', closeParamsIfOutside);
  return () => {
    clearInterval(id);
    document.removeEventListener('pointerdown', closeParamsIfOutside);
  };
});
</script>

<div class="page chat">
  <header>
    <h2>{$t('chat.title')}</h2>
  </header>

  <div class="transcript" bind:this={transcript} onscroll={onTranscriptScroll}>
    {#if messages.length === 0}
      <p class="muted">{$t('chat.empty')}</p>
    {/if}
    {#each messages as m, i (i)}
    <div class="bubble" class:user={m.role === 'user'}>
      <div class="role">{m.role}</div>
      {#if m.images?.length}
        <div class="thumbs">
          {#each m.images as img}
            <img src={img.url} alt={img.name} />
          {/each}
        </div>
      {/if}
      {#if m.role === 'assistant'}
        {@const think = splitThinking(m.content)}
        {#if think.has && think.thinking.trim()}
          {#if think.open}
            <div class="thinking">
              <div class="thinking-head">{$t('chat.thinking')}</div>
              <div class="thinking-body">{think.thinking}</div>
            </div>
          {:else}
            <details class="thinking">
              <summary>{$t('chat.thinking')}</summary>
              <div class="thinking-body">{think.thinking}</div>
            </details>
          {/if}
        {/if}
        <div class="content"
          >{think.answer}{#if busy && i === messages.length - 1}<span class="caret"></span>{/if}</div
        >
      {:else}
        <div class="content">{m.content}</div>
      {/if}
        {#if m.role === 'assistant' && m.stats}
          <div class="meta">
            {tps(m.stats)} tok/s · {m.stats.generated_tokens} tok · {$t('chat.firstToken')}
            {Math.round(m.stats.first_token_ms)} ms
            {#if m.stats.draft_tokens > 0}
              · {$t('chat.draftHit', {
                n: m.stats.draft_tokens,
                a: m.stats.accepted_draft_tokens,
                pct: Math.round((100 * m.stats.accepted_draft_tokens) / m.stats.draft_tokens),
              })}
            {/if}
          </div>
        {/if}
      </div>
    {/each}
  </div>

  {#if error}
    <div class="error">{error}</div>
  {/if}

  {#if models.length === 0 && apiOnly > 0}
    <div class="notice">{$t('chat.apiOnly', { n: apiOnly })}</div>
  {/if}

  <div class="composer">
    {#if images.length}
      <div class="attachments">
        {#each images as img, i}
          <span class="attachment">
            <img src={img.url} alt={img.name} />
            <button onclick={() => removeImage(i)} title={$t('chat.removeImage')} aria-label="remove">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round">
                <path d="M6 6l12 12M18 6L6 18" />
              </svg>
            </button>
          </span>
        {/each}
      </div>
    {/if}
    <textarea bind:value={input} onkeydown={onKeydown} placeholder={$t('chat.placeholder')} rows="3"
    ></textarea>
    <div class="bar">
      <input
        bind:this={fileInput}
        type="file"
        accept="image/*"
        multiple
        hidden
        onchange={(e) => addImages([...e.target.files]).catch(() => {})}
      />
      <button
        class="ghost attach"
        onclick={() => fileInput?.click()}
        disabled={busy}
        title={$t('chat.attachImage')}
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21.4 11.05l-8.5 8.5a5.5 5.5 0 01-7.78-7.78l8.5-8.5a3.67 3.67 0 015.18 5.18l-8.49 8.5a1.83 1.83 0 01-2.59-2.6l7.85-7.85" />
        </svg>
      </button>

      <select
        class="model"
        value={selected}
        onchange={(e) => rememberModel(e.target.value)}
        disabled={models.length === 0}
        title={$t('chat.model')}
      >
        {#if models.length === 0}
          <option value="">{$t('models.none')}</option>
        {/if}
        {#each models as m}
          <option value={m.id}
            >{m.id}{m.info?.vision ? ' · vision' : ''}{m.load_mtp ? ' · MTP' : m.draft_model ? ' · draft' : ''}</option
          >
        {/each}
      </select>

      <details class="params" bind:open={paramsOpen} bind:this={paramsEl}>
        <summary>{$t('chat.params')}</summary>
        <div class="params-panel">
          <label>{$t('chat.temp')} <input type="number" step="0.05" min="0" max="2" bind:value={temperature} /></label>
          <label>{$t('chat.topK')} <input type="number" step="1" min="0" bind:value={topK} /></label>
          <label>{$t('chat.topP')} <input type="number" step="0.05" min="0" max="1" bind:value={topP} /></label>
          <label>{$t('chat.minP')} <input type="number" step="0.01" min="0" max="1" bind:value={minP} /></label>
          <label
            >{$t('chat.repeatPenalty')} <input type="number" step="0.05" min="0" bind:value={repeatPenalty} /></label
          >
          <label
            >{$t('chat.repeatLastN')} <input type="number" step="1" min="0" bind:value={repeatLastN} /></label
          >
          <label>{$t('chat.maxtok')} <input type="number" step="1" min="0" bind:value={maxTokens} /></label>
          <label class="check" title={modelDrafts ? drafter : ''}>
            <input type="checkbox" bind:checked={speculative} />
            <span>{$t(modelDrafts ? 'chat.speculativeModel' : 'chat.speculative')}</span>
          </label>
          {#if speculative && !modelDrafts}
            <label>{$t('chat.ngramN')} <input type="number" step="1" min="1" max="16" bind:value={ngramN} /></label>
            <label>{$t('chat.ngramM')} <input type="number" step="1" min="1" max="32" bind:value={ngramM} /></label>
          {/if}
          <label>
            {$t('chat.seed')}
            <span class="seed">
              <select bind:value={seedMode}>
                <option value="random">{$t('chat.random')}</option>
                <option value="fixed">{$t('chat.fixed')}</option>
              </select>
              <input type="number" step="1" min="0" bind:value={seed} disabled={seedMode !== 'fixed'} />
            </span>
          </label>
        </div>
      </details>

      {#if busy}<span class="spinner" aria-hidden="true"></span>{/if}
      {#if busy}
        <button class="send" style="color:#e5484d" onclick={stop} title={$t('chat.stop')} aria-label={$t('chat.stop')}>
          <svg viewBox="0 0 24 24" fill="currentColor">
            <rect x="7" y="7" width="10" height="10" rx="1.5" />
          </svg>
        </button>
      {:else}
        <button class="send" onclick={send} disabled={busy || models.length === 0} title={$t('chat.send')}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 19V5M5 12l7-7 7 7" />
          </svg>
        </button>
      {/if}
    </div>
  </div>
</div>
