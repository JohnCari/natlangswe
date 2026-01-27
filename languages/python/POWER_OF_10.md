# The Power of 10 Rules for Safety-Critical Python Code

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

## The Power of 10 Rules — Python Edition

### Rule 1: Simple Control Flow — No Recursion

**Original Intent:** Eliminate complex control flow that impedes static analysis and can cause stack overflows.

**Python Adaptation:**

```python
# BAD: Recursion (stack overflow risk, hard to analyze)
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# GOOD: Iterative implementation
def factorial(n: int) -> int:
    assert n >= 0, "n must be non-negative"
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
```

**Guidelines:**
- Recursion is forbidden — Python has limited stack depth (~1000)
- Convert recursive algorithms to iterative with explicit stacks
- Limit `break`/`continue` to simple cases
- Avoid deeply nested conditionals (max 3-4 levels)

---

### Rule 2: Fixed Loop Bounds — Provable Termination

**Original Intent:** Ensure all loops terminate and can be analyzed statically.

**Python Adaptation:**

```python
from itertools import islice

# BAD: Unbounded iteration
for item in infinite_generator():
    process(item)

# BAD: While without clear bound
while condition:
    do_something()

# GOOD: Bounded with islice
MAX_ITEMS = 10_000
for item in islice(generator(), MAX_ITEMS):
    process(item)

# GOOD: Bounded with assertion helper
def iter_bounded(iterable, max_iter: int):
    for count, item in enumerate(iterable):
        assert count < max_iter, f"Exceeded {max_iter} iterations"
        yield item
```

**Guidelines:**
- All loops must have provable upper bounds
- Use `islice()` to enforce maximum iterations
- Define bounds as constants for clarity
- Prefer `for` over `while`

---

### Rule 3: Controlled Memory — Pagination and Chunking

**Original Intent:** Prevent unbounded memory growth and allocation failures.

**Python Adaptation:**

```python
# BAD: Load everything into memory
all_records = list(db.query("SELECT * FROM huge_table"))

# GOOD: Paginate/chunk data
PAGE_SIZE = 1000

def process_all_records():
    offset = 0
    while True:
        page = db.query(f"SELECT * FROM table LIMIT {PAGE_SIZE} OFFSET {offset}")
        if not page:
            break
        for record in page:
            process(record)
        offset += PAGE_SIZE
        assert offset < 10_000_000, "Safety limit exceeded"
```

**Guidelines:**
- Use generators for large data streams
- Paginate database queries
- Set explicit memory limits on collections
- Use `sys.getsizeof()` to monitor object sizes

---

### Rule 4: Function Length Limit — 60 Lines Maximum

**Original Intent:** Ensure functions are small enough to understand, test, and verify.

**Python Adaptation:**

```python
# BAD: Monolithic function
def process_order(order):
    # 200 lines doing everything...
    pass

# GOOD: Decomposed functions
def process_order(order: Order) -> Result:
    validate_order(order)
    total = calculate_totals(order)
    total = apply_discounts(order, total)
    save_order(order, total)
    return notify_customer(order)

def validate_order(order: Order) -> None:
    """≤60 lines"""
    ...

def calculate_totals(order: Order) -> Decimal:
    """≤60 lines"""
    ...
```

**Guidelines:**
- Maximum 60 lines per function
- Each function does one thing
- Use early returns to reduce nesting
- Extract complex conditionals into named functions

---

### Rule 5: Assertion Density — Minimum 2 Per Function

**Original Intent:** Defensive programming catches bugs early; assertions document invariants.

**Python Adaptation:**

```python
# BAD: No defensive checks
def transfer_funds(from_acct, to_acct, amount):
    from_acct.balance -= amount
    to_acct.balance += amount

# GOOD: Assert preconditions and postconditions
def transfer_funds(from_acct: Account, to_acct: Account, amount: Decimal) -> None:
    # Preconditions
    assert amount > 0, "Amount must be positive"
    assert from_acct.balance >= amount, "Insufficient funds"
    assert from_acct.id != to_acct.id, "Cannot transfer to same account"

    from_acct.balance -= amount
    to_acct.balance += amount

    # Postcondition
    assert from_acct.balance >= 0, "Balance went negative"
```

**Guidelines:**
- Minimum 2 assertions per non-trivial function
- Assert preconditions at function entry
- Assert postconditions before returning
- Use `assert` for invariants, exceptions for expected errors
- Keep assertions enabled in production for critical code

---

### Rule 6: Minimize Variable Scope

**Original Intent:** Reduce state complexity and potential for misuse.

**Python Adaptation:**

```python
# BAD: Global state
config = {}
def get_setting(key):
    return config[key]

# BAD: Variables declared far from use
def process():
    result = None
    temp = None
    # ... 50 lines ...
    result = compute()
    temp = transform(result)

# GOOD: Encapsulated in classes
class Config:
    def __init__(self):
        self._settings: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self._settings[key]

# GOOD: Variables at point of use
def process():
    result = compute()
    transformed = transform(result)
    return transformed
```

**Guidelines:**
- No global variables
- Use `_prefix` for private attributes
- Declare variables at first use
- Use classes/modules to encapsulate state
- Prefer immutable data where possible

---

### Rule 7: Check All Return Values

**Original Intent:** Never ignore errors; verify inputs at trust boundaries.

**Python Adaptation:**

```python
# BAD: Ignoring return value
open("file.txt")

# BAD: Bare except
try:
    risky_operation()
except:
    pass

# BAD: Swallowing exceptions
try:
    result = operation()
except Exception:
    result = None  # Silent failure

# GOOD: Handle or propagate
try:
    result = operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise

# GOOD: Validate at boundaries
def set_temperature(celsius: float) -> None:
    if not -273.15 <= celsius <= 1_000_000:
        raise ValueError(f"Temperature out of range: {celsius}")
    _internal_set_temp(celsius)
```

**Guidelines:**
- Never use bare `except:`
- Don't swallow exceptions silently
- Validate inputs at public API boundaries
- Let exceptions bubble up to proper handlers
- Use type hints to document expected returns

---

### Rule 8: Limit Metaprogramming

**Original Intent:** Avoid constructs that create unmaintainable, unanalyzable code.

**Python Adaptation:**

```python
# BAD: Dynamic code execution
eval(user_input)
exec(f"result = {expression}")

# BAD: Excessive decorator stacking
@log
@cache
@retry
@validate
@authorize
def my_function(): ...

# BAD: Magic methods everywhere
class Overengineered:
    def __getattr__(self, name): ...
    def __setattr__(self, name, value): ...
    def __call__(self, *args): ...

# GOOD: Simple, explicit code
def my_function():
    if not authorized():
        raise PermissionError()
    return compute_result()
```

**Guidelines:**
- Never use `eval()` or `exec()`
- Limit decorators to 1-2 per function
- Avoid metaclasses unless absolutely necessary
- Prefer explicit code over magic methods
- Keep `__getattr__`/`__setattr__` usage minimal

---

### Rule 9: Type Safety — Use Type Hints

**Original Intent:** (C: Restrict pointer usage for safety)

**Python Adaptation:**

```python
from typing import NamedTuple
from dataclasses import dataclass

# BAD: Untyped, mutable
def process(data):
    data["processed"] = True  # Mutates argument
    return data

# GOOD: Typed with immutable structures
class ProcessedData(NamedTuple):
    value: str
    processed: bool

def process(data: dict[str, Any]) -> ProcessedData:
    return ProcessedData(
        value=data["value"],
        processed=True
    )

# GOOD: Dataclass for structured data
@dataclass(frozen=True)
class Config:
    host: str
    port: int
    timeout: float = 30.0
```

**Guidelines:**
- Type hints on all function signatures
- Use `NamedTuple` or `@dataclass(frozen=True)` for data
- Don't mutate function arguments
- Use `typing.Final` for constants
- Prefer immutable collections (`tuple` over `list`)

---

### Rule 10: Static Analysis — Zero Warnings

**Original Intent:** Catch issues at development time; use every available tool.

**Python Adaptation:**

```toml
# pyproject.toml
[tool.mypy]
strict = true
warn_return_any = true
disallow_untyped_defs = true

[tool.ruff]
select = ["ALL"]
ignore = ["D203", "D213"]  # Docstring style conflicts

[tool.ruff.per-file-ignores]
"tests/*" = ["S101"]  # Allow assert in tests
```

```bash
# Required CI pipeline
mypy src/                    # Type checking
ruff check src/              # Linting
ruff format --check src/     # Formatting
pytest --cov=src tests/      # Tests with coverage
bandit -r src/               # Security analysis
```

**Guidelines:**
- Run mypy in strict mode
- Use ruff (or pylint + flake8)
- Zero warnings policy
- No `# type: ignore` without justification
- Run security scanners (bandit)

---

## Summary: Python Adaptation

| # | Original Rule | Python Guideline |
|---|---------------|------------------|
| 1 | No goto/recursion | No recursion, limit break/continue |
| 2 | Fixed loop bounds | Use `islice`, assertions, constants |
| 3 | No dynamic allocation | Paginate data, use generators |
| 4 | 60-line functions | Single responsibility, early returns |
| 5 | 2+ assertions/function | Assert pre/post conditions |
| 6 | Minimize scope | No globals, use classes, `_` prefix |
| 7 | Check returns | Handle exceptions, validate inputs |
| 8 | Limit preprocessor | No eval/exec, limit decorators |
| 9 | Restrict pointers | Type hints, immutable data |
| 10 | All warnings enabled | mypy strict, ruff, zero warnings |

---

## References

- [Original Power of 10 Paper](https://spinroot.com/gerard/pdf/P10.pdf) — Gerard Holzmann
- [10 Rules for Interpreted Languages](https://dev.to/xowap/10-rules-to-code-like-nasa-applied-to-interpreted-languages-40dd)
- [Powerof10-NASA GitHub](https://github.com/Vhivi/Powerof10-NASA)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [Ruff Linter](https://docs.astral.sh/ruff/)
