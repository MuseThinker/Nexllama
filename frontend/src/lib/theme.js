import { writable, get } from 'svelte/store';

const KEY = 'nexllama.theme';

function systemTheme() {
  if (typeof window === 'undefined') return 'dark';
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

function resolve(value) {
  return value === 'auto' ? systemTheme() : value;
}

function apply(value) {
  if (typeof document !== 'undefined') {
    document.documentElement.dataset.theme = resolve(value);
  }
}

export const theme = writable(localStorage.getItem(KEY) ?? 'auto');

theme.subscribe((value) => {
  localStorage.setItem(KEY, value);
  apply(value);
});

if (typeof window !== 'undefined') {
  window
    .matchMedia('(prefers-color-scheme: light)')
    .addEventListener('change', () => apply(get(theme)));
}
