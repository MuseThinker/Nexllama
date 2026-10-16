// Built-in cloud provider catalog: the direct model vendors, with the base URL
// to call and where a user gets a key. Non-secret, so it lives here rather than
// in settings. China and international accounts are separate entries because
// the base URL (and often the key) differs by region.
//
// `auth` lists the credential kinds a provider accepts:
//   "bearer"     — a normal API key, sent as `Authorization: Bearer`.
//   "token-plan" — a subscription/session token, pasted the same way.
//
// `console` is where to buy/obtain the key; opened in the system browser.

export const CLOUD_CATALOG = [
  // ── International direct vendors ─────────────────────────────────────────
  {
    id: 'openai',
    name: 'OpenAI',
    baseUrl: 'https://api.openai.com/v1',
    console: 'https://platform.openai.com/api-keys',
    auth: ['bearer'],
  },
  {
    id: 'anthropic',
    name: 'Anthropic Claude',
    baseUrl: 'https://api.anthropic.com/v1',
    console: 'https://console.anthropic.com/settings/keys',
    auth: ['bearer'],
  },
  {
    id: 'google',
    name: 'Google Gemini',
    baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai/',
    console: 'https://aistudio.google.com/app/apikey',
    auth: ['bearer'],
  },
  {
    id: 'xai',
    name: 'xAI Grok',
    baseUrl: 'https://api.x.ai/v1',
    console: 'https://console.x.ai/',
    auth: ['bearer'],
  },
  {
    id: 'xai-cli',
    name: 'xAI Grok（CLI 套餐 / token plan）',
    baseUrl: 'https://cli-chat-proxy.grok.com/v1',
    console: 'https://console.x.ai/',
    auth: ['token-plan'],
  },
  {
    id: 'mistral',
    name: 'Mistral',
    baseUrl: 'https://api.mistral.ai/v1',
    console: 'https://console.mistral.ai/api-keys/',
    auth: ['bearer'],
  },
  {
    id: 'cohere',
    name: 'Cohere',
    baseUrl: 'https://api.cohere.ai/compatibility/v1',
    console: 'https://dashboard.cohere.com/api-keys',
    auth: ['bearer'],
  },
  {
    id: 'deepseek',
    name: 'DeepSeek',
    baseUrl: 'https://api.deepseek.com',
    console: 'https://platform.deepseek.com/api_keys',
    auth: ['bearer'],
  },

  // ── 国产直连（中国 / 国际各一条）─────────────────────────────────────────
  {
    id: 'zhipu',
    name: '智谱 GLM（中国）',
    baseUrl: 'https://open.bigmodel.cn/api/paas/v4',
    console: 'https://open.bigmodel.cn/usercenter/apikeys',
    auth: ['bearer', 'token-plan'],
  },
  {
    id: 'zhipu-intl',
    name: '智谱 GLM（国际）',
    baseUrl: 'https://api.z.ai/api/paas/v4',
    console: 'https://z.ai/',
    auth: ['bearer'],
  },
  {
    id: 'kimi',
    name: '月之暗面 Kimi（中国）',
    baseUrl: 'https://api.moonshot.cn/v1',
    console: 'https://platform.moonshot.cn/',
    auth: ['bearer'],
  },
  {
    id: 'kimi-intl',
    name: '月之暗面 Kimi（国际）',
    baseUrl: 'https://api.moonshot.ai/v1',
    console: 'https://platform.moonshot.ai/',
    auth: ['bearer'],
  },
  {
    id: 'qwen',
    name: '阿里通义 Qwen（中国）',
    baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    console: 'https://bailian.console.aliyun.com/',
    auth: ['bearer'],
  },
  {
    id: 'qwen-intl',
    name: '阿里通义 Qwen（国际）',
    baseUrl: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
    console: 'https://modelstudio.console.alibabacloud.com/',
    auth: ['bearer'],
  },
  {
    id: 'minimax',
    name: 'MiniMax（中国）',
    baseUrl: 'https://api.minimaxi.com/v1',
    console: 'https://platform.minimaxi.com/',
    auth: ['bearer', 'token-plan'],
  },
  {
    id: 'minimax-intl',
    name: 'MiniMax（国际）',
    baseUrl: 'https://api.minimax.io/v1',
    console: 'https://platform.minimax.io/',
    auth: ['bearer'],
  },
];

export const catalogById = (id) => CLOUD_CATALOG.find((c) => c.id === id) ?? null;
