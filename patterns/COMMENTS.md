# Code Comments

> Inline, concise, and descriptive.

## Core Rules

1. **Always inline** — Comments go on the same line or directly above the code they describe
2. **Concise** — One line when possible, never rambling
3. **Descriptive** — Explain *why*, not *what* (the code shows what)

---

## Good Comments

```rust
let timeout = 30; // seconds, matches upstream API limit

// Retry 3x because network can be flaky on cold start
for _ in 0..3 {
    if try_connect().is_ok() { break; }
}
```

```python
# Skip first row (header)
for row in data[1:]:
    process(row)
```

---

## Bad Comments

```rust
// BAD: Describes what (obvious from code)
let x = x + 1; // increment x

// BAD: Too verbose
// This function takes a user ID as input and returns the user object
// from the database after validating that the ID is a valid UUID format
fn get_user(id: Uuid) -> User { ... }

// BAD: Outdated/wrong
let retries = 5; // retry twice
```

---

## When to Comment

- **Non-obvious business logic** — Why this specific value or condition
- **Workarounds** — Explain the bug or limitation being worked around
- **Performance choices** — Why this approach over the obvious one
- **External dependencies** — Links to docs, specs, or issue trackers

## When NOT to Comment

- Self-explanatory code
- Function/variable names that already describe intent
- Obvious operations
- TODOs without context (use issue tracker instead)

---

## Summary

- Inline, not block headers
- One line, not paragraphs
- Explain why, not what
- Delete comments that lie or clutter
