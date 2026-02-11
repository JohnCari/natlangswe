# The Power of 10 Rules for Safety-Critical SvelteKit Code

## Background

The **Power of 10 Rules** were created in 2006 by **Gerard J. Holzmann** at NASA's Jet Propulsion Laboratory (JPL) Laboratory for Reliable Software. These rules were designed for writing safety-critical code in C that could be effectively analyzed by static analysis tools.

The rules were incorporated into JPL's institutional coding standard and used for major missions including the **Mars Science Laboratory** (Curiosity Rover, 2012).

> *"If these rules seem draconian at first, bear in mind that they are meant to make it possible to check safety-critical code where human lives can very literally depend on its correctness."* — Gerard Holzmann

---

## The Original 10 Rules (C Language)

| # | Rule |
|---|------|
| 1 | Restrict all code to very simple control flow constructs—no `goto`, `setjmp`, `longjmp`, or recursion |
| 2 | Give all loops a fixed upper bound provable by static analysis |
| 3 | Do not use dynamic memory allocation after initialization |
| 4 | No function longer than one printed page (~60 lines) |
| 5 | Assertion density: minimum 2 assertions per function |
| 6 | Declare all data objects at the smallest possible scope |
| 7 | Check all return values and validate all function parameters |
| 8 | Limit preprocessor to header includes and simple macros |
| 9 | Restrict pointers: single dereference only, no function pointers |
| 10 | Compile with all warnings enabled; use static analyzers daily |

---

## The Power of 10 Rules — SvelteKit Edition

### Rule 1: Simple Control Flow — No Recursion, Guard Clauses in Load Functions

**Original Intent:** Eliminate complex control flow that impedes static analysis and can cause stack overflows.

**SvelteKit Adaptation:**

```typescript
// BAD: Recursive component (Svelte supports {#each} but not recursive self-reference safely)
<!-- RecursiveTree.svelte -->
<script lang="ts">
  import RecursiveTree from './RecursiveTree.svelte';

  let { node }: { node: TreeNode } = $props();
</script>

{node.label}
{#each node.children as child}
  <RecursiveTree node={child} />  <!-- Unbounded recursion -->
{/each}

// BAD: Deeply nested conditionals in load function
export const load: PageServerLoad = async ({ params, locals }) => {
  if (params.id) {
    const item = await db.get(params.id);
    if (item) {
      if (item.isPublished) {
        if (locals.user) {
          return { item };
        }
      }
    }
  }
};

// GOOD: Guard clauses with early returns in load functions
export const load: PageServerLoad = async ({ params, locals }) => {
  if (!params.id) {
    error(400, 'Missing ID');
  }

  const item = await db.get(params.id);
  if (!item) {
    error(404, 'Item not found');
  }

  if (!item.isPublished) {
    error(403, 'Item not published');
  }

  if (!locals.user) {
    redirect(303, '/login');
  }

  return { item };
};

// GOOD: Iterative tree rendering with bounded depth
<!-- TreeView.svelte -->
<script lang="ts">
  const MAX_DEPTH = 5;

  let { nodes }: { nodes: TreeNode[] } = $props();

  function flattenTree(items: TreeNode[], depth = 0): FlatNode[] {
    if (depth >= MAX_DEPTH) return [];

    const result: FlatNode[] = [];
    for (const node of items) {
      result.push({ ...node, depth });
      if (node.children) {
        result.push(...flattenTree(node.children, depth + 1));
      }
    }
    return result;
  }

  const flatNodes = $derived(flattenTree(nodes));
</script>

{#each flatNodes as node (node.id)}
  <div style:margin-left="{node.depth * 16}px">
    {node.label}
  </div>
{/each}
```

**Guidelines:**
- No recursive components — flatten trees iteratively with depth limits
- Use guard clauses with SvelteKit's `error()` and `redirect()` for early returns in load functions
- Maximum 3-4 levels of nesting in `{#if}` blocks
- Prefer flat component structures over deeply nested `{#if}/{:else}` chains

---

### Rule 2: Bounded Loops — Paginate Data, Limit Iterations

**Original Intent:** Ensure all loops terminate and can be analyzed statically.

**SvelteKit Adaptation:**

```typescript
// BAD: Unbounded data in load function
// +page.server.ts
export const load: PageServerLoad = async () => {
  const users = await db.user.findMany();  // Could be millions
  return { users };
};

// BAD: Unbounded {#each} in template
{#each items as item}
  <ItemCard {item} />  <!-- Could render 100,000 DOM nodes -->
{/each}

// GOOD: Paginated load function
// +page.server.ts
const PAGE_SIZE = 50;
const MAX_PAGES = 100;

export const load: PageServerLoad = async ({ url }) => {
  const page = Math.min(
    Math.max(1, Number(url.searchParams.get('page') ?? 1)),
    MAX_PAGES,
  );

  const users = await db.user.findMany({
    skip: (page - 1) * PAGE_SIZE,
    take: PAGE_SIZE,
  });

  const total = await db.user.count();

  return { users, page, totalPages: Math.ceil(total / PAGE_SIZE) };
};

// GOOD: Bounded {#each} with slice
<!-- +page.svelte -->
<script lang="ts">
  const MAX_VISIBLE = 100;

  let { data }: { data: PageData } = $props();
  const boundedUsers = $derived(data.users.slice(0, MAX_VISIBLE));
</script>

{#each boundedUsers as user (user.id)}
  <UserCard {user} />
{/each}

{#if data.users.length > MAX_VISIBLE}
  <p>{data.users.length - MAX_VISIBLE} more users not shown.</p>
{/if}

<Pagination current={data.page} total={data.totalPages} />
```

**Guidelines:**
- Always paginate database queries in load functions
- Use `.slice(0, MAX)` before `{#each}` for arrays of unknown size
- Define bounds as constants
- Validate pagination params from `url.searchParams` with clamping
- Assert collection sizes at boundaries

---

### Rule 3: Controlled Memory — Server Load First, Minimal Client State

**Original Intent:** Prevent unbounded memory growth and allocation failures.

**SvelteKit Adaptation:**

```typescript
// BAD: Fetching data client-side when server load works
<!-- +page.svelte -->
<script lang="ts">
  import { onMount } from 'svelte';

  let users = $state<User[]>([]);
  let posts = $state<Post[]>([]);
  let comments = $state<Comment[]>([]);

  onMount(async () => {
    const res = await fetch('/api/everything');
    const data = await res.json();
    users = data.users;      // All in client memory
    posts = data.posts;
    comments = data.comments;
  });
</script>

// BAD: Duplicating server data in client state
<script lang="ts">
  let { data }: { data: PageData } = $props();
  let users = $state(data.users);  // Duplicated in memory
</script>

// GOOD: Server load function (data stays serialized, not in JS heap)
// +page.server.ts
export const load: PageServerLoad = async () => {
  const users = await getRecentUsers(10);
  const posts = await getRecentPosts(10);

  return { users, posts };
};

// +page.svelte
<script lang="ts">
  let { data }: { data: PageData } = $props();
</script>

<UserList users={data.users} />
<PostList posts={data.posts} />

// GOOD: Minimal client state for interactivity only
<script lang="ts">
  let { data }: { data: PageData } = $props();

  let searchQuery = $state('');
  const filtered = $derived(
    data.users.filter(u =>
      u.name.toLowerCase().includes(searchQuery.toLowerCase())
    )
  );
</script>

<input bind:value={searchQuery} placeholder="Search users..." />

{#each filtered as user (user.id)}
  <UserCard {user} />
{/each}

// GOOD: Streaming with SvelteKit
// +page.server.ts
export const load: PageServerLoad = async () => {
  return {
    quickData: await getQuickData(),
    slowData: getSlowData(),  // Returns a Promise — streams to client
  };
};

// +page.svelte
<script lang="ts">
  let { data }: { data: PageData } = $props();
</script>

<QuickSection data={data.quickData} />

{#await data.slowData}
  <LoadingSkeleton />
{:then slowData}
  <SlowSection data={slowData} />
{:catch error}
  <ErrorMessage {error} />
{/await}
```

**Guidelines:**
- Default to `+page.server.ts` load functions — keep data on the server
- Use `$state` only for genuinely interactive client-side state
- Don't duplicate server data into client state — use `$derived` to transform `data` props
- Stream slow data with deferred Promises in load functions
- Avoid `onMount` + `fetch` when a server load function suffices

---

### Rule 4: Short Components — 60 Lines Maximum

**Original Intent:** Ensure functions are small enough to understand, test, and verify.

**SvelteKit Adaptation:**

```svelte
<!-- BAD: Monolithic page component -->
<!-- +page.svelte (300+ lines of mixed logic and markup) -->
<script lang="ts">
  let { data }: { data: PageData } = $props();
  // 20 state declarations...
  // 15 derived values...
  // 10 event handlers...
</script>

<!-- 200 lines of markup with complex conditionals -->

<!-- GOOD: Decomposed into focused components -->
<!-- +page.svelte -->
<script lang="ts">
  let { data }: { data: PageData } = $props();
</script>

<ProductHeader product={data.product} />
<ProductGallery images={data.product.images} />
<ProductDetails product={data.product} />
<AddToCart product={data.product} />
<RelatedProducts products={data.related} />
```

```typescript
// GOOD: Extract logic into separate .ts modules
// lib/services/cart.ts
export function calculateTotal(items: CartItem[]): number {
  console.assert(items.length > 0, 'Cart must not be empty');
  return items.reduce((sum, item) => sum + item.price * item.quantity, 0);
}

export function applyDiscount(total: number, code: string): number {
  console.assert(total >= 0, 'Total must be non-negative');
  const discount = DISCOUNT_CODES[code];
  if (!discount) return total;
  return total * (1 - discount.percentage);
}

// GOOD: Extract shared logic into reusable .svelte.ts files
// lib/state/search.svelte.ts
export function createSearch<T>(items: () => T[], key: (item: T) => string) {
  let query = $state('');

  const filtered = $derived(
    items().filter(item =>
      key(item).toLowerCase().includes(query.toLowerCase())
    )
  );

  return {
    get query() { return query; },
    set query(v: string) { query = v; },
    get filtered() { return filtered; },
  };
}
```

**Guidelines:**
- Maximum 60 lines per `.svelte` component (script + markup + style)
- Extract logic into `.ts` modules and `.svelte.ts` reactive modules
- One component per file
- Keep `+page.svelte` as a thin orchestrator composing child components
- Extract load function logic into service modules

---

### Rule 5: Validation — Zod for All Inputs, Validate at Boundaries

**Original Intent:** Defensive programming catches bugs early; assertions document invariants.

**SvelteKit Adaptation:**

```typescript
import { z } from 'zod';
import { error, fail } from '@sveltejs/kit';

// Define schemas for all external data
const UserSchema = z.object({
  id: z.string().uuid(),
  email: z.string().email(),
  name: z.string().min(1).max(100),
  role: z.enum(['admin', 'user']),
});

type User = z.infer<typeof UserSchema>;

// GOOD: Validate route params in load functions
// +page.server.ts
const ParamsSchema = z.object({
  id: z.string().uuid(),
});

export const load: PageServerLoad = async ({ params }) => {
  const result = ParamsSchema.safeParse(params);
  if (!result.success) {
    error(400, 'Invalid parameters');
  }

  const user = await getUser(result.data.id);
  if (!user) {
    error(404, 'User not found');
  }

  return { user };
};

// GOOD: Validate search params
const SearchParamsSchema = z.object({
  page: z.coerce.number().int().min(1).max(100).default(1),
  sort: z.enum(['name', 'date', 'score']).default('date'),
});

export const load: PageServerLoad = async ({ url }) => {
  const params = SearchParamsSchema.safeParse(
    Object.fromEntries(url.searchParams)
  );

  if (!params.success) {
    error(400, 'Invalid search parameters');
  }

  const items = await getItems(params.data);
  return { items, ...params.data };
};

// GOOD: Validate form actions with Zod
// +page.server.ts
const CreateUserSchema = z.object({
  email: z.string().email('Invalid email address'),
  name: z.string().min(1, 'Name is required').max(100),
  password: z.string().min(8, 'Password must be at least 8 characters'),
});

export const actions = {
  create: async ({ request }) => {
    const formData = await request.formData();
    const rawData = Object.fromEntries(formData);

    const result = CreateUserSchema.safeParse(rawData);
    if (!result.success) {
      return fail(400, {
        errors: result.error.flatten().fieldErrors,
        values: rawData,
      });
    }

    const user = await createUser(result.data);
    return { user };
  },
} satisfies Actions;

// GOOD: Validate API endpoint inputs
// +server.ts
const ApiInputSchema = z.object({
  query: z.string().min(1).max(200),
  limit: z.number().int().min(1).max(100).default(20),
});

export const POST: RequestHandler = async ({ request }) => {
  const body = await request.json();
  const result = ApiInputSchema.safeParse(body);

  if (!result.success) {
    return json({ error: result.error.flatten() }, { status: 400 });
  }

  const data = await search(result.data);
  return json({ data });
};
```

**Guidelines:**
- Validate ALL external inputs with Zod/Valibot
- Validate route `params` and `url.searchParams` in load functions
- Validate `formData` in form actions — return `fail()` with field errors
- Validate request bodies in `+server.ts` API endpoints
- Use `safeParse` — don't throw on invalid data
- Infer TypeScript types from schemas

---

### Rule 6: Minimal Scope — Server First, Colocate State

**Original Intent:** Reduce state complexity and potential for misuse.

**SvelteKit Adaptation:**

```typescript
// BAD: Global client store for server data
// lib/stores/users.ts
import { writable } from 'svelte/store';

export const users = writable<User[]>([]);  // Global mutable state!

// BAD: Top-level $state shared across components
// lib/state.svelte.ts
export let globalUsers = $state<User[]>([]);  // Mutable from anywhere

// BAD: Over-fetching in a single load function
export const load: PageServerLoad = async () => {
  return {
    users: await getUsers(),
    products: await getProducts(),
    orders: await getOrders(),
    analytics: await getAnalytics(),  // Does this page really need all of this?
  };
};

// GOOD: Server load function scoped to page needs
// routes/users/+page.server.ts
export const load: PageServerLoad = async ({ url }) => {
  const page = Number(url.searchParams.get('page') ?? 1);
  const users = await getUsers({ page, limit: 20 });
  return { users, page };
};

// GOOD: Layout load for shared data, page load for page-specific data
// routes/+layout.server.ts
export const load: LayoutServerLoad = async ({ locals }) => {
  return { user: locals.user };  // Only truly shared data
};

// GOOD: Colocated client state
<!-- SearchableList.svelte -->
<script lang="ts">
  let { items }: { items: Item[] } = $props();

  let query = $state('');  // Local to this component only
  const filtered = $derived(
    items.filter(item =>
      item.name.toLowerCase().includes(query.toLowerCase())
    )
  );
</script>

<input bind:value={query} />

{#each filtered as item (item.id)}
  <ItemCard {item} />
{/each}

// GOOD: Scoped reactive state with .svelte.ts when sharing is needed
// lib/state/cart.svelte.ts
class CartState {
  items = $state<CartItem[]>([]);

  get total() {
    return this.items.reduce((sum, item) => sum + item.price * item.quantity, 0);
  }

  add(item: CartItem) {
    console.assert(item.quantity > 0, 'Quantity must be positive');
    this.items.push(item);
  }

  remove(id: string) {
    this.items = this.items.filter(item => item.id !== id);
  }
}

// Instantiate in a layout, pass down via context
export function createCart() {
  return new CartState();
}
```

**Guidelines:**
- Default to `+page.server.ts` load functions — avoid client-side data fetching
- Colocate `$state` with the component that uses it
- Use layout load functions only for data genuinely shared across routes
- Avoid global writable stores for server data — use load functions
- Use Svelte context (`setContext`/`getContext`) for shared client state within a subtree
- Keep client boundaries as small as possible

---

### Rule 7: Check Returns — Handle All States in Load and Actions

**Original Intent:** Never ignore errors; verify inputs at trust boundaries.

**SvelteKit Adaptation:**

```typescript
// BAD: Unchecked database result in load
export const load: PageServerLoad = async ({ params }) => {
  const user = await getUser(params.id);
  return { user };  // user could be null — template will crash
};

// BAD: Ignoring form action result
<!-- +page.svelte -->
<form method="POST" use:enhance>
  <button>Submit</button>
  <!-- No error handling -->
</form>

// GOOD: Handle null/error in load functions
export const load: PageServerLoad = async ({ params }) => {
  const user = await getUser(params.id);
  if (!user) {
    error(404, 'User not found');
  }

  return { user };
};

// GOOD: Handle all form action states
<!-- +page.svelte -->
<script lang="ts">
  import { enhance } from '$app/forms';

  let { form }: { form: ActionData } = $props();
</script>

{#if form?.errors}
  <div class="error">
    {#each Object.entries(form.errors) as [field, messages]}
      <p>{field}: {messages?.join(', ')}</p>
    {/each}
  </div>
{/if}

{#if form?.success}
  <div class="success">Operation completed successfully.</div>
{/if}

<form method="POST" action="?/create" use:enhance>
  <label>
    Email
    <input name="email" value={form?.values?.email ?? ''} />
  </label>
  <button>Create</button>
</form>

// GOOD: Handle all async states in templates
<!-- +page.svelte -->
<script lang="ts">
  let { data }: { data: PageData } = $props();
</script>

{#await data.slowData}
  <LoadingSkeleton />
{:then result}
  {#if result.items.length === 0}
    <EmptyState message="No items found" />
  {:else}
    {#each result.items as item (item.id)}
      <ItemCard {item} />
    {/each}
  {/if}
{:catch err}
  <ErrorMessage message={err.message} />
{/await}

// GOOD: Typed action results
// +page.server.ts
export const actions = {
  update: async ({ request, params }) => {
    const formData = await request.formData();
    const result = UpdateSchema.safeParse(Object.fromEntries(formData));

    if (!result.success) {
      return fail(400, {
        success: false as const,
        errors: result.error.flatten().fieldErrors,
        values: Object.fromEntries(formData),
      });
    }

    try {
      await updateItem(params.id, result.data);
    } catch (e) {
      return fail(500, {
        success: false as const,
        errors: { _global: ['Update failed. Please try again.'] },
        values: Object.fromEntries(formData),
      });
    }

    return { success: true as const };
  },
} satisfies Actions;
```

**Guidelines:**
- Always handle `null`/`undefined` returns in load functions with `error()`
- Handle `form` action data in templates — show errors and preserve values
- Handle loading, error, and empty states with `{#await}` blocks
- Return typed results from form actions using `fail()` for errors
- Never ignore Promise rejections in `+server.ts` endpoints

---

### Rule 8: Limit Metaprogramming — No {@html}, No eval

**Original Intent:** Avoid constructs that create unmaintainable, unanalyzable code.

**SvelteKit Adaptation:**

```svelte
<!-- BAD: XSS vulnerability with {@html} -->
<script lang="ts">
  let { data }: { data: PageData } = $props();
</script>

<div>{@html data.userContent}</div>  <!-- User-controlled HTML! -->

<!-- BAD: Dynamic component construction -->
<script lang="ts">
  const componentMap: Record<string, any> = {};

  // Dynamically importing unknown components
  const Component = componentMap[data.type];
</script>

<svelte:component this={Component} />

<!-- BAD: eval or new Function -->
<script lang="ts">
  function runUserCode(code: string) {
    eval(code);  // Never!
  }
</script>

<!-- GOOD: Sanitize if HTML is absolutely required -->
<script lang="ts">
  import DOMPurify from 'isomorphic-dompurify';

  let { html }: { html: string } = $props();

  const sanitized = $derived(
    DOMPurify.sanitize(html, {
      ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'p', 'br', 'a'],
      ALLOWED_ATTR: ['href', 'target', 'rel'],
    })
  );
</script>

<div>{@html sanitized}</div>

<!-- GOOD: Use Svelte components instead of HTML strings -->
<script lang="ts">
  let { text, bold }: { text: string; bold: boolean } = $props();
</script>

{#if bold}
  <strong>{text}</strong>
{:else}
  <span>{text}</span>
{/if}

<!-- GOOD: Explicit component mapping with known set -->
<script lang="ts">
  import Alert from './Alert.svelte';
  import Info from './Info.svelte';
  import Warning from './Warning.svelte';

  const components = { alert: Alert, info: Info, warning: Warning } as const;

  let { type }: { type: keyof typeof components } = $props();

  const Component = $derived(components[type]);
</script>

<Component />

<!-- GOOD: Composition over complex logic in templates -->
<script lang="ts">
  import type { Snippet } from 'svelte';

  let { header, content, footer }: {
    header: Snippet;
    content: Snippet;
    footer?: Snippet;
  } = $props();
</script>

<div class="card">
  <header>{@render header()}</header>
  <main>{@render content()}</main>
  {#if footer}
    <footer>{@render footer()}</footer>
  {/if}
</div>
```

**Guidelines:**
- **Never use `{@html}` with user content**
- If HTML is required, sanitize with DOMPurify (whitelist only)
- Never use `eval()` or `new Function()`
- Use explicit component maps with typed keys instead of dynamic imports
- Prefer Svelte snippets and composition over complex metaprogramming
- Avoid `<svelte:component>` with untyped/unknown component references

---

### Rule 9: Type Safety — Strict TypeScript, Typed Load Functions

**Original Intent:** (C: Restrict pointer usage for safety)

**SvelteKit Adaptation:**

```typescript
// BAD: Using any in load functions
export const load: PageServerLoad = async ({ params }: any) => {
  const data = await fetch('/api/data');
  return data.json();  // Untyped response
};

// BAD: Untyped form action
export const actions = {
  create: async ({ request }: any) => {
    const data = await request.formData();
    // No validation, no types...
  },
};

// BAD: Type assertions without validation
const user = data as User;  // Dangerous assumption

// GOOD: Fully typed load functions with Zod inference
import type { PageServerLoad } from './$types';

const ItemSchema = z.object({
  id: z.string().uuid(),
  title: z.string(),
  status: z.enum(['draft', 'published', 'archived']),
});

type Item = z.infer<typeof ItemSchema>;

export const load: PageServerLoad = async ({ params }) => {
  const raw = await db.item.findUnique({ where: { id: params.id } });
  if (!raw) {
    error(404, 'Item not found');
  }

  const item = ItemSchema.parse(raw);
  return { item };
};

// GOOD: Typed form actions with satisfies
import type { Actions } from './$types';

const CreateItemSchema = z.object({
  title: z.string().min(1).max(200),
  status: z.enum(['draft', 'published']),
});

export const actions = {
  create: async ({ request, locals }) => {
    if (!locals.user) {
      error(401, 'Unauthorized');
    }

    const formData = await request.formData();
    const result = CreateItemSchema.safeParse(Object.fromEntries(formData));

    if (!result.success) {
      return fail(400, { errors: result.error.flatten().fieldErrors });
    }

    const item = await createItem(result.data, locals.user.id);
    return { item };
  },
} satisfies Actions;

// GOOD: Typed API endpoints
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ url, locals }) => {
  if (!locals.user) {
    return json({ error: 'Unauthorized' }, { status: 401 });
  }

  const query = url.searchParams.get('q') ?? '';
  const results = await search(query);
  return json({ results });
};

// GOOD: Branded types for IDs
type UserId = string & { readonly brand: unique symbol };
type ItemId = string & { readonly brand: unique symbol };

function getItem(userId: UserId, itemId: ItemId) {
  // Can't accidentally swap userId and itemId
}
```

**svelte.config.js:**
```javascript
import adapter from '@sveltejs/adapter-auto';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

export default {
  preprocess: vitePreprocess(),
  kit: {
    adapter: adapter(),
    typescript: {
      config: (config) => ({
        ...config,
        compilerOptions: {
          ...config.compilerOptions,
          strict: true,
          noImplicitAny: true,
          strictNullChecks: true,
          noImplicitReturns: true,
          noFallthroughCasesInSwitch: true,
          noUncheckedIndexedAccess: true,
        },
      }),
    },
  },
};
```

**Guidelines:**
- Always import types from `./$types` — SvelteKit auto-generates them
- Enable all strict TypeScript options
- Never use `any` — use `unknown` and validate
- Infer types from Zod schemas
- Use `satisfies Actions` for type-safe form actions
- Use branded types for IDs
- Type all `+server.ts` handlers with `RequestHandler`

---

### Rule 10: Static Analysis — svelte-check, ESLint, Zero Warnings

**Original Intent:** Catch issues at development time; use every available tool.

**SvelteKit Adaptation:**

```javascript
// eslint.config.js
import js from '@eslint/js';
import ts from 'typescript-eslint';
import svelte from 'eslint-plugin-svelte';
import svelteParser from 'svelte-eslint-parser';

export default ts.config(
  js.configs.recommended,
  ...ts.configs.strictTypeChecked,
  ...svelte.configs['flat/recommended'],
  {
    files: ['**/*.svelte', '**/*.svelte.ts'],
    languageOptions: {
      parser: svelteParser,
      parserOptions: {
        parser: ts.parser,
      },
    },
  },
  {
    rules: {
      // Security
      'no-eval': 'error',

      // TypeScript strict
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/no-non-null-assertion': 'error',
      '@typescript-eslint/no-floating-promises': 'error',

      // Svelte
      'svelte/no-at-html-tags': 'error',
      'svelte/require-each-key': 'error',
      'svelte/no-dom-manipulating': 'error',
      'svelte/no-reactive-reassign': 'error',
    },
  },
);
```

```bash
# Required CI pipeline
svelte-check --tsconfig ./tsconfig.json   # Svelte + TypeScript checking
eslint --max-warnings 0 src/              # Zero warnings
prettier --check src/                     # Formatting
vitest run                                # Tests
npm audit                                 # Dependency vulnerabilities
```

**Security Hooks (src/hooks.server.ts):**
```typescript
import type { Handle } from '@sveltejs/kit';

export const handle: Handle = async ({ event, resolve }) => {
  const response = await resolve(event);

  response.headers.set('X-Content-Type-Options', 'nosniff');
  response.headers.set('X-Frame-Options', 'DENY');
  response.headers.set('X-XSS-Protection', '1; mode=block');
  response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');
  response.headers.set(
    'Permissions-Policy',
    'camera=(), microphone=(), geolocation=()',
  );

  return response;
};
```

**Guidelines:**
- Run `svelte-check` in CI — it catches Svelte template + TypeScript errors
- Enable `svelte/no-at-html-tags` to flag `{@html}` usage
- Zero warnings policy
- Configure security headers in `hooks.server.ts`
- Run `npm audit` regularly
- Use Dependabot or Renovate for dependency updates

---

## Summary: SvelteKit Adaptation

| # | Original Rule | SvelteKit Guideline |
|---|---------------|---------------------|
| 1 | No goto/recursion | No recursive components, guard clauses with `error()`/`redirect()` |
| 2 | Fixed loop bounds | Paginate in load functions, `.slice()` before `{#each}` |
| 3 | No dynamic allocation | Server load first, minimize `$state`, stream with Promises |
| 4 | 60-line functions | 60-line components, extract to `.ts` and `.svelte.ts` modules |
| 5 | 2+ assertions/function | Zod validation for params, form actions, and API inputs |
| 6 | Minimize scope | Server load for data, colocate `$state`, context for shared state |
| 7 | Check returns | Handle `null` with `error()`, handle form errors, `{#await}` blocks |
| 8 | Limit preprocessor | No `{@html}` with user content, no eval, explicit component maps |
| 9 | Restrict pointers | Strict TS, `./$types` imports, no `any`, branded types |
| 10 | All warnings enabled | `svelte-check`, ESLint, zero warnings, security hooks |

---

## References

- [Original Power of 10 Paper](https://spinroot.com/gerard/pdf/P10.pdf) — Gerard Holzmann
- [SvelteKit Documentation](https://svelte.dev/docs/kit) — Official Docs
- [SvelteKit Security](https://svelte.dev/docs/kit/security) — Content Security Policy, CSRF
- [Svelte 5 Runes](https://svelte.dev/docs/svelte/what-are-runes) — $state, $derived, $effect
- [eslint-plugin-svelte](https://sveltejs.github.io/eslint-plugin-svelte/) — Svelte-specific linting
- [svelte-check](https://www.npmjs.com/package/svelte-check) — Type checking for Svelte
- [Zod Documentation](https://zod.dev/)
