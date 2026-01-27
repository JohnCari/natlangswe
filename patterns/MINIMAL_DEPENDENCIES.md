# Minimal Dependencies Philosophy

> The best dependency is no dependency. The second best is one you fully understand.

## Core Principles

1. **Only use libraries that are absolutely necessary**
2. **Prefer debuggability over convenience**
3. **The fewer dependencies, the better**
4. **Well-maintained and supported libraries only**

---

## When to Add a Library

Add a library when:

- **Complexity is high** — Solving the problem yourself would require significant expertise (crypto, compression, image processing)
- **Security is critical** — Authentication, encryption, and security-sensitive code should use battle-tested libraries
- **Standards compliance** — Protocol implementations (HTTP, WebSocket, TLS) that must be spec-compliant
- **Platform abstraction** — Cross-platform code that handles OS differences

### Good Library Indicators

- Active maintenance (commits in last 6 months)
- Stable API with semantic versioning
- Minimal transitive dependencies
- Good documentation
- Active community/issue responses
- Used by other major projects

---

## When to Avoid a Library

Avoid a library when:

- **You can write it in 2-3 lines** — Don't add `left-pad` for string padding
- **It's a "convenience" wrapper** — Thin wrappers over standard library functions
- **It adds many transitive dependencies** — Each dependency is a potential vulnerability
- **It's poorly maintained** — No updates, unresolved issues, abandoned
- **It makes debugging harder** — Abstractions that hide what's actually happening
- **It's overkill** — Using a full ORM when raw SQL would suffice

### Red Flags

- No commits in 12+ months
- Hundreds of open issues
- Breaking changes frequently
- Deep dependency trees
- No clear documentation
- Single maintainer with no succession plan

---

## Evaluation Checklist

Before adding a dependency, answer these questions:

```
[ ] Can I implement this in <50 lines of code?
[ ] Is this library actively maintained?
[ ] How many transitive dependencies does it add?
[ ] Can I easily debug through this library?
[ ] What happens if this library is abandoned?
[ ] Does this library have security vulnerabilities?
[ ] Is the license compatible with my project?
[ ] Do I understand what this library actually does?
```

If you answered "no" to any of these, reconsider.

---

## The Debugging Test

> If something breaks at 2 AM, can you fix it?

When evaluating a library, ask:

1. Can you read and understand its source code?
2. Can you step through it in a debugger?
3. Can you patch it locally if needed?
4. Can you replace it if the maintainer disappears?

If the answer is "no" to most of these, you're adding risk.

---

## Language-Specific Notes

### Rust
- Prefer `std` over external crates when possible
- Check `cargo tree` for dependency depth
- Use `cargo audit` for security vulnerabilities

### Python
- Prefer standard library (`json`, `pathlib`, `dataclasses`)
- Check `pip show -v <package>` for dependencies
- Use `pip-audit` for security checks

### TypeScript
- Prefer native APIs over polyfills
- Check `npm ls` for dependency tree
- Use `npm audit` regularly

---

## Example: When NOT to Add a Library

**Task:** Check if a string is empty or whitespace

```python
# BAD: Adding a library
from stringutils import is_blank
if is_blank(text):
    ...

# GOOD: 1 line of code
if not text or not text.strip():
    ...
```

**Task:** Deep clone an object

```typescript
// BAD: Adding lodash for one function
import { cloneDeep } from 'lodash';
const copy = cloneDeep(obj);

// GOOD: Native solution
const copy = structuredClone(obj);
```

---

## Summary

- Every dependency is a liability
- Write simple code instead of importing complexity
- When you must depend, depend wisely
- Understand what you import
- Be ready to replace or remove any dependency
