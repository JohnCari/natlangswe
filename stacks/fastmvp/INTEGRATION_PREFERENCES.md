# FastMVP Integration Preferences

> Opinionated defaults for integrating the FastMVP stack

---

## Backend Choice

| Choose | When |
|--------|------|
| **Convex** | Real-time features, collaborative apps, reactive queries |
| **Supabase** | Raw SQL needed, complex joins, PostgreSQL extensions (PostGIS, pg_vector) |

- Default to Convex unless you have a specific reason for Supabase
- Don't mix both in the same project — pick one

---

## Authentication (Clerk)

- Place middleware in `middleware.ts` at project root, not in layout
- Use `auth()` in Server Components, `useAuth()` in Client Components
- Store user metadata (preferences, settings) in your database, not Clerk
  - Keeps Clerk swappable if you ever migrate
  - Clerk is for auth, your DB is for user data
- Sync Clerk user creation to your DB via webhooks

```typescript
// middleware.ts
import { clerkMiddleware } from '@clerk/nextjs/server'
export default clerkMiddleware()
export const config = { matcher: ['/((?!.*\\..*|_next).*)', '/'] }
```

---

## Payments (Stripe)

- **Webhooks over polling** — never poll for payment status
- Create Checkout Sessions from API routes, not client-side
- Store `stripe_customer_id` in your user table on first purchase
- Use Stripe's hosted Checkout — don't build custom payment forms

```typescript
// app/api/webhooks/stripe/route.ts
export async function POST(req: Request) {
  const event = await verifyStripeWebhook(req)

  switch (event.type) {
    case 'checkout.session.completed':
      await handleSuccessfulPayment(event.data.object)
      break
    case 'customer.subscription.deleted':
      await handleCancellation(event.data.object)
      break
  }

  return new Response('OK')
}
```

---

## Environment Variables

- All secrets in `.env.local`, never committed
- Use `NEXT_PUBLIC_` prefix **only** for truly public values (publishable keys)
- Validate env vars at build time:

```typescript
// env.ts
import { createEnv } from '@t3-oss/env-nextjs'
import { z } from 'zod'

export const env = createEnv({
  server: {
    CLERK_SECRET_KEY: z.string(),
    STRIPE_SECRET_KEY: z.string(),
    STRIPE_WEBHOOK_SECRET: z.string(),
    CONVEX_DEPLOY_KEY: z.string().optional(),
  },
  client: {
    NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY: z.string(),
    NEXT_PUBLIC_CONVEX_URL: z.string().url(),
  },
  runtimeEnv: {
    // Map to process.env
  },
})
```

---

## File Structure

```
app/
├── api/
│   ├── webhooks/
│   │   ├── clerk/route.ts    # User sync webhook
│   │   └── stripe/route.ts   # Payment webhooks
│   └── stripe/
│       └── checkout/route.ts # Create Checkout Session
├── (auth)/
│   ├── sign-in/
│   └── sign-up/
└── (dashboard)/
    └── ...

convex/
├── _generated/
├── users.ts        # User-related functions
├── subscriptions.ts
└── schema.ts
```

---

## Provider Hierarchy

Wrap providers in this order (outermost to innermost):

```tsx
// app/layout.tsx
<ClerkProvider>
  <ConvexProviderWithClerk>
    <ThemeProvider>
      {children}
    </ThemeProvider>
  </ConvexProviderWithClerk>
</ClerkProvider>
```

---

## See Also

- [NextJS Power of 10](../../frameworks/nextjs/POWER_OF_10.md)
- [NextJS Preferences](../../frameworks/nextjs/PREFERENCES.md)
- [React Power of 10](../../frameworks/react/POWER_OF_10.md)
- [TypeScript Power of 10](../../languages/typescript/POWER_OF_10.md)
