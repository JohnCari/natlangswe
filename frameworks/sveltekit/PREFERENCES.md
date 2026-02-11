# SvelteKit Monorepo Preferences

> Optimized defaults for SvelteKit monorepo development on Vercel

## Core Stack

| Tool | Purpose |
|------|---------|
| **SvelteKit (TypeScript)** | Latest stable, Svelte 5 Runes |
| **Bun** | Package manager + runtime |
| **Turborepo** | Monorepo orchestration |
| **Vercel Adapter** | Deployment + Edge Functions |
| **Biome** | Linter + formatter |
| **Svelte MCP** | Official Svelte MCP server for AI tooling |

## Package Manager Config

Ensure Vercel CI uses Bun by specifying in root `package.json`:

```json
{
  "packageManager": "bun@1.1.0"
}
```

## Svelte 5 Runes — Always

**Always use Svelte 5 Runes.** Never use legacy Svelte 4 reactivity (`$:`, `export let`, writable stores for component state).

### Runes Cheat Sheet

| Legacy (Never Use) | Rune (Always Use) |
|---------------------|-------------------|
| `export let prop` | `let { prop } = $props()` |
| `$: derived = x + y` | `const derived = $derived(x + y)` |
| `$: { sideEffect() }` | `$effect(() => { sideEffect() })` |
| `let count = 0` (reactive) | `let count = $state(0)` |
| `<slot />` | `{@render children()}` with Snippets |

### Props — Always `$props()`

```svelte
<!-- GOOD: Svelte 5 Runes -->
<script lang="ts">
  import type { Snippet } from 'svelte';

  let {
    title,
    count = 0,
    children,
    onclick,
  }: {
    title: string;
    count?: number;
    children: Snippet;
    onclick: (e: MouseEvent) => void;
  } = $props();
</script>

<h1>{title} ({count})</h1>
<button {onclick}>Click</button>
{@render children()}
```

### State — Always `$state` and `$derived`

```svelte
<script lang="ts">
  let count = $state(0);
  let items = $state<string[]>([]);
  const doubled = $derived(count * 2);
  const total = $derived(items.length);
</script>
```

### Effects — Always `$effect`

```svelte
<script lang="ts">
  let query = $state('');

  $effect(() => {
    // Runs when query changes
    console.log('Search:', query);
  });
</script>
```

### Shared Reactive State — `.svelte.ts` Files

```typescript
// lib/state/counter.svelte.ts
export function createCounter(initial = 0) {
  let count = $state(initial);

  return {
    get count() { return count; },
    increment() { count++; },
    decrement() { count--; },
    reset() { count = initial; },
  };
}
```

```svelte
<!-- Usage in component -->
<script lang="ts">
  import { createCounter } from '$lib/state/counter.svelte';

  const counter = createCounter(10);
</script>

<button onclick={counter.increment}>{counter.count}</button>
```

### Class-Based State

```typescript
// lib/state/cart.svelte.ts
export class CartState {
  items = $state<CartItem[]>([]);

  get total() {
    return this.items.reduce((sum, i) => sum + i.price * i.quantity, 0);
  }

  get isEmpty() {
    return this.items.length === 0;
  }

  add(item: CartItem) {
    this.items.push(item);
  }

  remove(id: string) {
    this.items = this.items.filter(i => i.id !== id);
  }
}
```

## Server vs Client — Server First

**Default to server-side.** Only add client interactivity when necessary.

### When to Use Server Load (`+page.server.ts`)
- Database queries
- API calls with secrets
- Authentication checks
- Data that doesn't need client-side refetching

### When to Use Universal Load (`+page.ts`)
- Data that needs to run on both server and client (e.g., during client-side navigation)
- Public API calls without secrets

### When to Use Client State (`$state`)
- Form inputs and UI toggles
- Search/filter on existing data
- Animations and transitions
- Real-time updates via WebSocket

### Prevent Server Code Leakage

SvelteKit enforces this by convention — `+page.server.ts` and `+server.ts` files never ship to the client. For shared modules:

```typescript
// lib/server/db.ts
// Files under lib/server/ should only be imported from server files
import { DATABASE_URL } from '$env/static/private';  // Build error if imported client-side
```

## Vercel Adapter — Always

Always use `@sveltejs/adapter-vercel` for deployment:

```bash
bun add -D @sveltejs/adapter-vercel
```

```javascript
// svelte.config.js
import adapter from '@sveltejs/adapter-vercel';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

export default {
  preprocess: vitePreprocess(),
  kit: {
    adapter: adapter({
      runtime: 'nodejs22.x',
    }),
  },
};
```

### Edge Functions

For latency-sensitive routes, opt into Edge Runtime:

```typescript
// routes/api/fast/+server.ts
export const config = {
  runtime: 'edge',
};

export const GET: RequestHandler = async () => {
  return json({ fast: true });
};
```

## Tooling

### Biome — Always

Biome replaces both ESLint and Prettier. 10-20x faster.

```bash
bun add -D @biomejs/biome
bunx biome init
```

```json
// biome.json
{
  "$schema": "https://biomejs.dev/schemas/1.9.0/schema.json",
  "organizeImports": {
    "enabled": true
  },
  "linter": {
    "enabled": true,
    "rules": {
      "recommended": true,
      "suspicious": {
        "noExplicitAny": "error"
      },
      "complexity": {
        "noExcessiveCognitiveComplexity": "error"
      },
      "security": {
        "noGlobalEval": "error"
      }
    }
  },
  "formatter": {
    "enabled": true,
    "indentStyle": "tab",
    "lineWidth": 100
  },
  "files": {
    "ignore": [
      ".svelte-kit",
      "build",
      "node_modules"
    ]
  }
}
```

**Note:** Biome does not natively lint `.svelte` files. Use `svelte-check` alongside Biome for Svelte template validation:

```bash
# CI pipeline
bunx svelte-check --tsconfig ./tsconfig.json   # Svelte + TypeScript
bunx biome check src/                           # Lint + format TS/JS files
```

### Svelte MCP — Always

Always configure the official Svelte MCP server for AI-assisted development. This provides AI tools with up-to-date Svelte/SvelteKit documentation and APIs.

```json
// .mcp.json (project root)
{
  "mcpServers": {
    "svelte": {
      "command": "npx",
      "args": ["-y", "svelte-mcp"]
    }
  }
}
```

This ensures AI assistants have access to:
- Current SvelteKit API documentation
- Svelte 5 Runes syntax and patterns
- Component authoring best practices
- SvelteKit routing and load function patterns

### Turborepo Remote Caching

Free and automatic on Vercel. Enable with:

```bash
bunx turbo login
bunx turbo link
```

## Monorepo Structure

```
my-monorepo/
├── apps/
│   ├── web/                 # Main SvelteKit app
│   └── docs/                # Documentation site
├── packages/
│   ├── ui/                  # Shared Svelte components
│   ├── typescript-config/   # Shared tsconfig.json
│   └── biome-config/        # Shared Biome config
├── .mcp.json                # Svelte MCP server config
├── turbo.json
├── package.json             # Include "packageManager": "bun@1.1.0"
└── bun.lockb
```

### turbo.json

```json
{
  "$schema": "https://turbo.build/schema.json",
  "tasks": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": [".svelte-kit/**", "build/**"]
    },
    "dev": {
      "cache": false,
      "persistent": true
    },
    "check": {
      "dependsOn": ["^build"]
    },
    "lint": {}
  }
}
```

### Shared UI Package

```
packages/ui/
├── src/
│   ├── Button.svelte
│   ├── Card.svelte
│   └── index.ts
├── package.json
└── svelte.config.js
```

```json
// packages/ui/package.json
{
  "name": "@repo/ui",
  "type": "module",
  "svelte": "./src/index.ts",
  "exports": {
    ".": {
      "types": "./src/index.ts",
      "svelte": "./src/index.ts"
    }
  }
}
```

Usage in apps:

```svelte
<script lang="ts">
  import { Button, Card } from '@repo/ui';
</script>

<Card>
  <Button onclick={() => alert('clicked')}>Click me</Button>
</Card>
```

## Quick Start

```bash
# Create SvelteKit project with Bun
bun create svelte@latest my-app
cd my-app
bun install

# Add Vercel adapter
bun add -D @sveltejs/adapter-vercel

# Add Biome
bun add -D @biomejs/biome
bunx biome init

# For monorepo setup
bunx create-turbo@latest -m bun
# Then replace Next.js apps with SvelteKit apps
```

## References

- [SvelteKit Documentation](https://svelte.dev/docs/kit)
- [Svelte 5 Runes](https://svelte.dev/docs/svelte/what-are-runes)
- [SvelteKit Vercel Adapter](https://svelte.dev/docs/kit/adapter-vercel)
- [Turborepo + SvelteKit](https://turborepo.dev/docs/guides/frameworks/sveltekit)
- [Biome Documentation](https://biomejs.dev/guides/getting-started/)
- [Svelte MCP Server](https://github.com/nichochar/svelte-mcp)
- [Bun Package Manager](https://bun.sh/docs/cli/install)
