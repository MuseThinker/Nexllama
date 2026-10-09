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
  } catch {
    online = false;
  } finally {
    probing = false;
  }
}

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
