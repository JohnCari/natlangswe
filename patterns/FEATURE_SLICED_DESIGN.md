# Feature Sliced Design for Monorepos

> Structure by purpose, not by type. Every feature owns its slice.

## Core Principles

1. **Layered architecture** — Code is organized into strict layers with clear responsibilities
2. **Unidirectional imports** — A layer can only import from layers below it, never above or sideways
3. **Slices are isolated** — Each slice is self-contained and exposes only a public API
4. **Monorepo = layers as packages** — Each layer (or group of slices) becomes its own package

---

## Layers

From top (most specific) to bottom (most shared):

```
app/          → Wiring, configuration, entry points
  features/   → User-facing functionality (each feature is a slice)
  entities/   → Business objects and their operations
  shared/     → Reusable utilities, UI primitives, constants
```

Each layer has a single responsibility:

| Layer      | Owns                          | Example                          |
|------------|-------------------------------|----------------------------------|
| `app`      | Routing, providers, startup   | App config, global error handler |
| `features` | Use cases, user interactions  | Login flow, checkout, search     |
| `entities` | Domain models, business rules | User, Product, Order             |
| `shared`   | Generic helpers, primitives   | Formatters, validators, logger   |

---

## Monorepo Structure

In a monorepo, layers map to packages:

```
monorepo/
├── packages/
│   ├── app/                  # Entry points, wiring
│   │   ├── src/
│   │   └── package config
│   ├── features/
│   │   ├── auth/             # Slice: authentication
│   │   │   ├── src/
│   │   │   └── public-api    # Only exports what others need
│   │   ├── checkout/         # Slice: checkout flow
│   │   │   ├── src/
│   │   │   └── public-api
│   │   └── search/           # Slice: search
│   │       ├── src/
│   │       └── public-api
│   ├── entities/
│   │   ├── user/             # Slice: user model
│   │   ├── product/          # Slice: product model
│   │   └── order/            # Slice: order model
│   └── shared/
│       ├── lib/              # Utilities, helpers
│       └── config/           # Shared constants
└── workspace config
```

---

## Import Rules

The only rule that matters:

```
app → features → entities → shared
 ↓        ↓          ↓
 OK       OK         OK      (import from layers below)
 ✗        ✗          ✗       (import from layers above or same layer)
```

### Good Imports

```
# Feature imports an entity — allowed (layer below)
features/checkout → entities/order
features/auth     → shared/lib

# Entity imports shared — allowed (layer below)
entities/user → shared/lib
```

### Bad Imports

```
# BAD: Entity imports a feature (layer above)
entities/user → features/auth

# BAD: Feature imports another feature (same layer)
features/checkout → features/auth

# BAD: Shared imports anything above it
shared/lib → entities/user
```

---

## Slice Rules

Each slice (a folder within a layer) follows these rules:

1. **Public API** — Every slice exposes a single entry point (index file, manifest, barrel export). Internal files are private.
2. **Self-contained** — A slice owns its models, logic, and tests. No reaching into another slice's internals.
3. **No cross-slice imports within the same layer** — `features/auth` cannot import from `features/checkout`. If both need something, it belongs in a lower layer.

### Good Slice

```
features/checkout/
├── src/
│   ├── calculate-total       # Internal logic
│   ├── validate-cart         # Internal logic
│   └── apply-discount        # Internal logic
├── tests/
│   └── checkout-tests
└── public-api                # Exports only: processCheckout, CheckoutResult
```

Consumers only see `processCheckout` and `CheckoutResult`. Everything else is private.

### Bad Slice

```
# BAD: No public API, internals exposed
features/checkout/
├── calculate-total           # Imported directly by other features
├── validate-cart             # Imported directly by app
└── apply-discount            # Imported by entities (layer violation)
```

---

## When to Split a Slice

Split when:

- **A slice has multiple unrelated responsibilities** — If `features/user` handles registration, profile editing, and password reset, split into `features/register`, `features/profile`, `features/password-reset`
- **A slice grows beyond a few files** — Large slices are a sign of mixed concerns
- **Two teams need to work on the same slice independently** — Slice boundaries align with team boundaries

Don't split when:

- The logic is tightly coupled and changes together
- Splitting would create circular dependencies
- It's only a few functions

---

## When Features Need to Communicate

Features on the same layer cannot import each other. Use these patterns instead:

- **Move shared logic down** — If two features need the same thing, it belongs in `entities` or `shared`
- **Event-based communication** — Features emit events that other features subscribe to, decoupled through the `app` layer
- **Orchestration in app** — The `app` layer wires features together

---

## Summary

- Organize by layers: `app → features → entities → shared`
- Import only downward, never upward or sideways
- Each slice has a public API — internals stay private
- In a monorepo, layers are packages
- When two features need each other, push shared logic down
- Split slices by responsibility, not by size
