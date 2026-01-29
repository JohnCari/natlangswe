# NextJS Monorepo Preferences

> Optimized defaults for NextJS monorepo development on Vercel

## Core Stack

| Tool | Purpose |
|------|---------|
| **NextJS (TSX)** | Latest stable, App Router |
| **Bun** | Package manager + runtime |
| **Turbopack** | Dev bundler (default in Next.js) |
| **Turborepo** | Monorepo orchestration |
| **Vercel** | Hosting + CI/CD |

## Package Manager Config

Ensure Vercel CI uses Bun by specifying in root `package.json`:

```json
{
  "packageManager": "bun@1.1.0"
}
```

## Server Components

**Default to React Server Components (RSC).** Only use Client Components when absolutely necessary.

### When to Use RSC (Default)
- Data fetching
- Static content
- Database queries
- Backend logic
- Components without interactivity

### When to Use Client Components (`'use client'`)
- Event handlers (onClick, onChange)
- useState, useEffect, useReducer
- Browser APIs (localStorage, window)
- Real-time updates
- Animations and transitions

### Prevent Code Leakage

Install boundary packages:
```bash
bun add server-only client-only
```

Use in server-only files:
```tsx
import 'server-only'  // Build error if imported client-side

export async function getSecretData() {
  // This code can never leak to the client
}
```

### Client Boundary Placement

Keep client boundaries **as high as possible** in the component tree:

```tsx
// GOOD: Client boundary at the leaf
// app/page.tsx (Server Component)
export default function Page() {
  const data = await fetchData()  // Server-side fetch
  return (
    <div>
      <StaticHeader />           {/* Server Component */}
      <InteractiveButton />      {/* Client Component - leaf */}
    </div>
  )
}

// BAD: Client boundary too low (wraps everything)
'use client'
export default function Page() { ... }
```

**Push providers deep:**
```tsx
// GOOD: Provider wraps only what needs it
<ServerLayout>
  <ThemeProvider>        {/* Client boundary here */}
    <InteractiveContent />
  </ThemeProvider>
</ServerLayout>
```

### Container/Presentational Pattern

Keep data-fetching on the server, pass to presentational client components:

```tsx
// ProductContainer.tsx (Server Component)
export async function ProductContainer({ id }: { id: string }) {
  const product = await db.product.find(id)  // Never ships to browser
  return <ProductCard product={product} />   // Client presentational
}
```

### Streaming with Suspense

Always wrap async components or use `loading.tsx` for streaming:

```tsx
// app/dashboard/loading.tsx
export default function Loading() {
  return <DashboardSkeleton />
}

// Or use Suspense directly
<Suspense fallback={<Skeleton />}>
  <AsyncComponent />
</Suspense>
```

Without Suspense boundaries, React treats the entire app as one chunk.

## Edge Runtime

For latency-sensitive routes, use Edge Runtime:

```tsx
// app/api/fast/route.ts
export const runtime = 'edge'

export async function GET() {
  return Response.json({ fast: true })
}
```

## Tooling

### Biome (Recommended)
Faster linter + formatter than ESLint + Prettier combined (10-20x faster).

```bash
bun add -D @biomejs/biome
bunx biome init
```

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
│   ├── web/                 # Main NextJS app
│   └── docs/                # Documentation site (Nextra)
├── packages/
│   ├── ui/                  # Shared Shadcn components
│   ├── typescript-config/   # Shared tsconfig.json
│   └── biome-config/        # Shared Biome config
├── turbo.json
├── package.json             # Include "packageManager": "bun@1.1.0"
└── bun.lockb
```

## Quick Start

```bash
# Create monorepo with Turborepo + Bun
bunx create-turbo@latest -m bun

# Or clone a starter
git clone https://github.com/gmickel/turborepo-shadcn-nextjs
cd turborepo-shadcn-nextjs
bun install
bun dev
```

## References

- [Next.js Server and Client Components](https://nextjs.org/docs/app/getting-started/server-and-client-components)
- [Turbopack API Reference](https://nextjs.org/docs/app/api-reference/turbopack)
- [Turborepo + Next.js Guide](https://turborepo.dev/docs/guides/frameworks/nextjs)
- [Biome Documentation](https://biomejs.dev/guides/getting-started/)
- [Bun-powered Turborepo Starter](https://github.com/gmickel/turborepo-shadcn-nextjs)
- [React Server Components Patterns](https://www.patterns.dev/react/react-server-components/)
