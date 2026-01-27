# Lisa Flow

> Structured AI development with automatic TDD and self-healing tests

**Repository:** [github.com/JohnCari/lisa-flow](https://github.com/JohnCari/lisa-flow)

---

## Workflow

```
SPECIFY → PLAN → TASKS → IMPLEMENT → TEST LOOP
```

| Phase | Description |
|-------|-------------|
| **Specify** | Creates spec with TDD requirement (automatic) |
| **Plan** | Technical implementation plan |
| **Tasks** | Breaks down into tasks.md |
| **Implement** | Builds the feature |
| **Test Loop** | Runs tests, fixes failures, repeats until pass |

---

## Usage

```bash
./lisa-flow.sh "feature description" [max_test_iterations]
```

Tests are automatically included — no need to specify "with tests".

---

## Integration with Feature-Sliced Hexagonal

Lisa Flow is the **inner workflow** each terminal runs within [Feature-Sliced Hexagonal](../patterns/FEATURE_SLICED_HEXAGONAL.md):

```
1. IDEATION     → Describe idea
2. ANALYSIS     → Create skeleton if needed
3. BREAKDOWN    → Split into features, assign terminals
4. PARALLEL     → Each terminal runs lisa-flow.sh
                  └── SPECIFY → PLAN → TASKS → IMPLEMENT → TEST LOOP
5. INTEGRATION  → Merge all features
```

- Feature-Sliced Hexagonal prevents conflicts (file ownership)
- Lisa Flow gives each terminal a structured TDD workflow

---

## Inspiration

| Source | Contribution |
|--------|--------------|
| [Ralph Loop](https://ghuntley.com/loop/) | Self-healing test loop |
| [GitHub Spec Kit](https://github.com/github/spec-kit) | Structured spec phases |
