<script>
import { onDestroy } from 'svelte';
import { daemon } from './lib/api.js';
import { t } from './lib/i18n.js';
import Chat from './routes/Chat.svelte';
import Models from './routes/Models.svelte';
import Settings from './routes/Settings.svelte';
import wordmark from './assets/logo-wordmark.png';

const tabs = [
  { id: 'chat', key: 'nav.chat' },
  { id: 'models', key: 'nav.models' },
  { id: 'settings', key: 'nav.settings' },
];

let tab = $state('chat');
let online = $state(false);

// The daemon answered and then refused the calls behind the token. `/health` is
// deliberately open so a client can bootstrap before it has one, which means
// this cannot come from the probe: it comes from any panel's first real call,
// and it is raised as one event by `api.js` rather than caught in each page.
let needsToken = $state(false);

// One probe at a time. Without the guard, a daemon that takes longer than the
// interval to answer stacks requests until the browser runs out of connections
// for this host and nothing else can get through either.
let probing = false;

async function probe() {
  if (probing) return;
  probing = true;
  try {
    await daemon.health();
    online = true;
    if (needsToken) {
      // The banner says "paste your token". The first call that gets through
      // with it is the proof, so the banner takes itself down rather than
      // waiting to be dismissed — and it only disappears once it is true.
      try {
        await daemon.status();
        needsToken = false;
      } catch {
        /* still the wrong token, or the daemon went away: leave it up */
      }
    }
  } catch {
    online = false;
  } finally {
    probing = false;
  }
}

$effect(() => {
  const unauthorized = () => (needsToken = true);
  window.addEventListener('nexllama:unauthorized', unauthorized);
  return () => window.removeEventListener('nexllama:unauthorized', unauthorized);
});

probe();
const poll = setInterval(probe, 5000);
onDestroy(() => clearInterval(poll));
</script>

<style>
  /* `display: contents` keeps the wrapper out of the layout, so hiding a tab
     costs nothing but the `display: none` below — but the component inside it
     stays mounted, which is the point: `Chat` owns the transcript, and
     unmounting it on every tab change used to throw the conversation away. */
  .pane {
    display: contents;
  }
  .pane.hidden {
    display: none;
  }

  /* The one thing the app cannot work around on its own: the daemon is up, so
     the status dot is green, and every call behind the token is refused. */
  .banner {
    display: flex;
    gap: 0.75rem;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 1rem;
    background: color-mix(in srgb, var(--warn, #b45309) 18%, transparent);
    border-bottom: 1px solid var(--line);
    font-size: 0.85rem;
  }
</style>

<div class="app">
  <aside>
    <img class="brand" src={wordmark} alt="Nexllama" />
    <nav>
      {#each tabs as item}
        <button class:active={tab === item.id} onclick={() => (tab = item.id)}
          >{$t(item.key)}</button
        >
      {/each}
    </nav>
    <div class="status" class:on={online}>
      <span class="dot"></span>
      {online ? $t('status.online') : $t('status.offline')}
    </div>
    <div class="addr">{daemon.base}</div>
  </aside>

  <main>
    {#if needsToken}
      <div class="banner">
        <span>{$t('auth.tokenNeeded')}</span>
        <button onclick={() => (tab = 'settings')}>{$t('nav.settings')}</button>
      </div>
    {/if}
    <div class="pane" class:hidden={tab !== 'chat'}>
      <Chat />
    </div>
    <div class="pane" class:hidden={tab !== 'models'}>
      <Models />
    </div>
    <div class="pane" class:hidden={tab !== 'settings'}>
      <Settings />
    </div>
  </main>
</div>
