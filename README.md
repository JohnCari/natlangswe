# NatLangSWE

> A Natural Language Software Developer — Best practices for AI-assisted development to prevent vibe coding

## What is NatLangSWE?

**NatLangSWE** stands for **Natural Language Software Engineer** — a discipline for writing software through structured communication with AI coding assistants.

Natural Language Software Development is the practice of directing AI assistants like Claude Code using explicit, well-defined guidelines rather than ad-hoc prompts. This repository teaches the best practices for this emerging development paradigm.

## Why NatLangSWE?

### The Problem: Vibe Coding

"Vibe coding" happens when developers prompt AI assistants without clear constraints or standards, resulting in:

- Inconsistent code quality
- Security vulnerabilities
- Poor architectural decisions
- Code that "works" but violates best practices
- Technical debt that compounds with each generation

### The Solution: Structured Natural Language Development

Instead of vague prompts like *"write me a web server"*, Natural Language Software Development provides explicit rules that AI assistants follow. By adding these guidelines to your AI's context, you get code that adheres to proven standards — whether that's NASA's safety-critical rules, idiomatic language patterns, or framework-specific conventions.

## How to Use with Claude Code

1. **Clone natlangswe** into your project (or as a sibling directory):
   ```bash
   cd your-project
   git clone https://github.com/NatLangSWE/natlangswe.git natlangswe
   ```
2. **Delete what you don't need** — Remove the files and folders that aren't relevant to your stack (e.g., delete `frameworks/pytorch/` if you're not using PyTorch)
3. **Create a `CLAUDE.md`** in your project root
4. **Reference the remaining files** in your `CLAUDE.md` so Claude Code reads them before writing code

### Example `CLAUDE.md`

For a SvelteKit + TypeScript + Bun + PostgreSQL project:

```markdown
# Coding Standards

Read the relevant natlangswe/ files before writing code.

## Reference Files

- natlangswe/languages/typescript/POWER_OF_10.md
- natlangswe/frameworks/sveltekit/POWER_OF_10.md
- natlangswe/frameworks/svelte5-runes/POWER_OF_10.md
- natlangswe/frameworks/tailwindcss/POWER_OF_10.md
- natlangswe/databases/postgresql/POWER_OF_10.md
- natlangswe/toolchains/bun/POWER_OF_10.md
- natlangswe/frameworks/sveltekit/PREFERENCES.md
- natlangswe/patterns/COMMENTS.md
- natlangswe/patterns/MINIMAL_DEPENDENCIES.md
- natlangswe/patterns/FEATURE_SLICED_DESIGN.md
```

See [stacks/svelteship/CLAUDE.md](stacks/svelteship/CLAUDE.md) for a full working example.

## Repository Structure

```
NatLangSWE/
├── languages/
│   ├── rust/       # Rust guidelines
│   ├── python/     # Python guidelines
│   └── typescript/ # TypeScript guidelines
├── frameworks/
│   ├── nextjs/         # NextJS (TSX) guidelines
│   ├── sveltekit/      # SvelteKit (TypeScript) guidelines
│   ├── svelte5-runes/  # Svelte 5 Runes (reactivity) guidelines
│   ├── react/          # React (TSX) guidelines
│   ├── tailwindcss/    # TailwindCSS guidelines
│   ├── axum/           # Rust Axum guidelines
│   ├── electron/       # Electron (TypeScript) guidelines
│   ├── fastapi/        # FastAPI (Python) guidelines
│   └── pytorch/        # PyTorch (Python) guidelines
├── databases/
│   └── postgresql/     # PostgreSQL guidelines
├── toolchains/
│   ├── uv/             # UV (Python toolchain) guidelines
│   ├── bun/            # Bun (TypeScript toolchain) guidelines
│   └── cargo-rustup/   # Cargo + Rustup (Rust toolchain) guidelines
├── patterns/       # Cross-language design patterns
├── stacks/         # Recommended tech stacks
```

---

## Guidelines

### Power of 10 Rules

NASA JPL's safety-critical coding rules adapted for modern development.

| Language | Guidelines |
|----------|------------|
| Rust | [POWER_OF_10.md](languages/rust/POWER_OF_10.md) |
| Python | [POWER_OF_10.md](languages/python/POWER_OF_10.md) |
| TypeScript | [POWER_OF_10.md](languages/typescript/POWER_OF_10.md) |

| Framework | Guidelines |
|-----------|------------|
| React | [POWER_OF_10.md](frameworks/react/POWER_OF_10.md) |
| NextJS | [POWER_OF_10.md](frameworks/nextjs/POWER_OF_10.md) ・ [PREFERENCES.md](frameworks/nextjs/PREFERENCES.md) |
| SvelteKit | [POWER_OF_10.md](frameworks/sveltekit/POWER_OF_10.md) ・ [PREFERENCES.md](frameworks/sveltekit/PREFERENCES.md) |
| Svelte 5 Runes | [POWER_OF_10.md](frameworks/svelte5-runes/POWER_OF_10.md) |
| TailwindCSS | [POWER_OF_10.md](frameworks/tailwindcss/POWER_OF_10.md) |
| Electron | [POWER_OF_10.md](frameworks/electron/POWER_OF_10.md) |
| Axum | [POWER_OF_10.md](frameworks/axum/POWER_OF_10.md) |
| FastAPI | [POWER_OF_10.md](frameworks/fastapi/POWER_OF_10.md) |
| PyTorch | [POWER_OF_10.md](frameworks/pytorch/POWER_OF_10.md) |

| Database | Guidelines |
|----------|------------|
| PostgreSQL | [POWER_OF_10.md](databases/postgresql/POWER_OF_10.md) |

| Toolchain | Guidelines |
|-----------|------------|
| UV | [POWER_OF_10.md](toolchains/uv/POWER_OF_10.md) |
| Bun | [POWER_OF_10.md](toolchains/bun/POWER_OF_10.md) |
| Cargo + Rustup | [POWER_OF_10.md](toolchains/cargo-rustup/POWER_OF_10.md) |

### Patterns

- [Feature-Sliced Design](patterns/FEATURE_SLICED_DESIGN.md) — Monorepo architecture with strict layers
- [Minimal Dependencies](patterns/MINIMAL_DEPENDENCIES.md) — When to use vs. avoid libraries
- [Code Comments](patterns/COMMENTS.md) — Inline, concise commenting guidelines

### Stacks

- [FastMVP](stacks/fastmvp/README.md) — Vercel + NextJS + Convex/Supabase + Clerk + Stripe
- [SvelteShip](stacks/svelteship/README.md) — Vercel + SvelteKit + Supabase + Stripe

---

## Contributing

Contributions welcome! Feel free to submit guidelines for additional languages or improvements to existing ones.

## License

CC0 1.0 Universal — Public Domain
