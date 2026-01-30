# FastMVP Stack

> Ship your MVP in days, not months

---

## Contents

- [The Stack](#the-stack)
- [Why This Stack?](#why-this-stack)
- [Server Components](#server-components)
- [Convex vs Supabase](#convex-vs-supabase)
- [When to Use FastMVP](#when-to-use-fastmvp)
- [Quick Start](#quick-start)
- [Resources](#resources)

---

## What is FastMVP?

FastMVP is a curated tech stack optimized for **Minimum Viable Products** (MVP) and **Minimum Evolvable Products** (MEP). Every component is chosen for developer velocity, seamless integration, and generous free tiers.

---

## The Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Hosting** | [Vercel](https://vercel.com) | Zero-config deployment, edge functions, preview URLs |
| **Frontend** | [Next.js](https://nextjs.org) | App Router, Server Components, API routes |
| **Styling** | [Tailwind CSS](https://tailwindcss.com) | Utility-first CSS, minimal bundle size |
| **Components** | [shadcn/ui](https://ui.shadcn.com) | Copy-paste components you own, accessible defaults |
| **Backend** | [Convex](https://convex.dev) / [Supabase](https://supabase.com) | Real-time database, serverless functions |
| **Auth** | [Clerk](https://clerk.com) | Drop-in authentication, user management |
| **Payments** | [Stripe](https://stripe.com/checkout) | Hosted checkout, subscriptions |
| **Tooling** | [Bun](https://bun.sh) + [Turborepo](https://turborepo.dev) | Fast runtime, monorepo orchestration |
| **Linting** | [Biome](https://biomejs.dev) | Linter + formatter (10-20x faster than ESLint) |

---

## Why This Stack?

### Seamless Integration

| Integration | Benefit |
|-------------|---------|
| Vercel + Next.js | First-party support, automatic optimizations |
| Clerk + Next.js | Official SDK, middleware, Server Components |
| Convex/Supabase + Clerk | Built-in auth adapters |
| Stripe Checkout | No custom payment UI needed |

### Developer Experience

- TypeScript end-to-end
- Hot reload everywhere
- Type-safe database queries
- Preview deployments on every PR

### Cost-Effective

| Service | Free Tier |
|---------|-----------|
| Vercel | Generous hobby tier |
| Convex | Handles most MVP traffic |
| Supabase | 500MB database |
| Clerk | 10,000 MAUs |
| Stripe | Pay only on transactions |

---

## Server Components

**Default to React Server Components (RSC).** Only add `'use client'` when necessary.

| Use RSC (Default) | Use Client Component |
|-------------------|---------------------|
| Data fetching | Event handlers (`onClick`) |
| Static content | `useState`, `useEffect` |
| Database queries | Browser APIs (`localStorage`) |
| Backend logic | Real-time subscriptions |

> Keep client boundaries high in the tree — push providers deep, keep interactive components as leaves.

See [NextJS Preferences](../../frameworks/nextjs/PREFERENCES.md) for detailed RSC guidelines.

---

## Convex vs Supabase

| Factor | Convex | Supabase |
|--------|--------|----------|
| **Best for** | Real-time, collaborative apps | CRUD, PostgreSQL familiarity |
| **Database** | Document-based, reactive | PostgreSQL (relational) |
| **Functions** | TypeScript on their infra | Edge Functions (Deno) |
| **Real-time** | Built-in, automatic | Requires setup |
| **Learning curve** | New paradigm | Familiar SQL |

**Choose Convex** → Real-time features, reactive queries, collaborative apps

**Choose Supabase** → Complex relational data, raw SQL, PostgreSQL extensions

---

## When to Use FastMVP

### Use It When

- Validating a product idea quickly
- Building a SaaS with auth + payments
- Solo developer or small team
- You need to ship in weeks

### Consider Alternatives When

- Enterprise compliance (SOC2, HIPAA)
- Heavy backend processing
- Offline-first mobile apps
- Full infrastructure control needed

---

## Quick Start

### 1. Create Monorepo

```bash
bunx create-turbo@latest -m bun
```

### 2. Add UI

```bash
bunx shadcn@latest init
```

### 3. Add Linting

```bash
bun add -D @biomejs/biome && bunx biome init
```

### 4. Deploy

Connect GitHub repo to [Vercel](https://vercel.com)

### 5. Add Auth

```bash
bun add @clerk/nextjs
```

Wrap app in `<ClerkProvider>`, add middleware

### 6. Add Backend

**Convex:**
```bash
bun add convex && bunx convex dev
```

**Supabase:**
```bash
bun add @supabase/supabase-js
```

### 7. Add Payments

Create Stripe Checkout Session from API route

### 8. Ship It

---

## See Also

- [Integration Preferences](./INTEGRATION_PREFERENCES.md) — Opinionated integration patterns
- [NextJS Power of 10](../../frameworks/nextjs/POWER_OF_10.md) — Safety-critical coding rules
- [NextJS Preferences](../../frameworks/nextjs/PREFERENCES.md) — Monorepo configuration
- [React Power of 10](../../frameworks/react/POWER_OF_10.md) — Client-side React patterns
- [TypeScript Power of 10](../../languages/typescript/POWER_OF_10.md) — TypeScript best practices

---

## Resources

| Category | Links |
|----------|-------|
| **Framework** | [Next.js](https://nextjs.org/docs) ・ [Tailwind](https://tailwindcss.com/docs) ・ [shadcn/ui](https://ui.shadcn.com/docs) |
| **Backend** | [Convex](https://docs.convex.dev) ・ [Supabase](https://supabase.com/docs) |
| **Services** | [Clerk](https://clerk.com/docs) ・ [Stripe](https://stripe.com/docs/checkout) ・ [Vercel](https://vercel.com/docs) |
| **Tooling** | [Bun](https://bun.sh/docs) ・ [Turborepo](https://turborepo.dev/docs) ・ [Biome](https://biomejs.dev) |
