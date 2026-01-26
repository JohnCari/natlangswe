# The Power of 10 Rules for Safety-Critical Rust Code

## Background

The **Power of 10 Rules** were created in 2006 by **Gerard J. Holzmann** at NASA's Jet Propulsion Laboratory (JPL) Laboratory for Reliable Software. These rules were designed for writing safety-critical code in C that could be effectively analyzed by static analysis tools.

The rules were incorporated into JPL's institutional coding standard and used for major missions including the **Mars Science Laboratory** (Curiosity Rover, 2012), which had over 3 million lines of flight software.

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

## The Power of 10 Rules — Rust Edition

### Rule 1: Simple Control Flow — No Recursion

**Original Intent:** Eliminate complex control flow that impedes static analysis and can cause stack overflows.

**Rust Adaptation:**

```rust
// ❌ FORBIDDEN: Direct or indirect recursion
fn factorial(n: u64) -> u64 {
    if n <= 1 { 1 } else { n * factorial(n - 1) }  // Stack overflow risk
}

// ✅ REQUIRED: Iterative implementation with bounded loops
fn factorial(n: u64) -> Option<u64> {
    if n > 20 { return None; }  // Prevent overflow
    let mut result: u64 = 1;
    for i in 2..=n {
        result = result.checked_mul(i)?;
    }
    Some(result)
}
```

**Rust-Specific Guidelines:**
- **Recursion is forbidden** — Rust does not guarantee tail-call optimization
- Convert all recursive algorithms to iterative form using explicit stacks
- If recursion is unavoidable, use the `#[recursive]` crate with bounded depth
- Avoid closures that capture and call themselves
- `goto` doesn't exist in Rust ✓
- `setjmp`/`longjmp` don't exist in Rust ✓

**Clippy Enforcement:**
```toml
# clippy.toml
cognitive-complexity-threshold = 15
```

---

### Rule 2: Fixed Loop Bounds — Provable Termination

**Original Intent:** Ensure all loops terminate and can be analyzed statically.

**Rust Adaptation:**

```rust
// ❌ FORBIDDEN: Unbounded loops
while condition {  // Cannot prove termination
    // ...
}

loop {  // Infinite loop
    if should_break { break; }
}

// ❌ FORBIDDEN: Iterator without bounds
for item in unbounded_iterator {
    // ...
}

// ✅ REQUIRED: Fixed bounds with explicit limits
const MAX_ITERATIONS: usize = 10_000;

for i in 0..MAX_ITERATIONS {
    if done { break; }
    // ...
}

// ✅ REQUIRED: Bounded iterators
for item in collection.iter().take(MAX_ITEMS) {
    // ...
}

// ✅ REQUIRED: Explicit bounds in retry logic
for attempt in 0..MAX_RETRIES {
    match operation() {
        Ok(result) => return Ok(result),
        Err(e) if attempt == MAX_RETRIES - 1 => return Err(e),
        Err(_) => continue,
    }
}
```

**Rust-Specific Guidelines:**
- All `loop`, `while`, and `for` constructs must have provable upper bounds
- Use `.take(N)` on iterators to enforce maximum iterations
- Define maximum bounds as `const` values for static verification
- Document the rationale for bound selection
- Prefer `for` over `while` and `loop`

**Clippy Enforcement:**
```rust
#![deny(clippy::infinite_loop)]
#![deny(clippy::maybe_infinite_iter)]
```

---

### Rule 3: No Dynamic Memory Allocation After Initialization

**Original Intent:** Prevent heap fragmentation, allocation failures, and unpredictable latency.

**Rust Adaptation:**

```rust
// ❌ FORBIDDEN: Runtime heap allocation
fn process_data(input: &[u8]) -> Vec<u8> {
    let mut buffer = Vec::new();  // Heap allocation
    buffer.extend_from_slice(input);
    buffer
}

// ❌ FORBIDDEN: Box, Rc, Arc in runtime code
let data = Box::new(LargeStruct::default());

// ✅ REQUIRED: Stack-allocated fixed-size arrays
fn process_data(input: &[u8]) -> [u8; MAX_BUFFER_SIZE] {
    let mut buffer = [0u8; MAX_BUFFER_SIZE];
    let len = input.len().min(MAX_BUFFER_SIZE);
    buffer[..len].copy_from_slice(&input[..len]);
    buffer
}

// ✅ REQUIRED: Use `heapless` crate for no_std collections
use heapless::Vec;
const CAPACITY: usize = 1024;

let mut buffer: Vec<u8, CAPACITY> = Vec::new();
buffer.push(42).ok();  // Returns Err if full
```

**Rust-Specific Guidelines:**
- Use `#![no_std]` to eliminate standard library heap usage
- Replace `Vec<T>` with `heapless::Vec<T, N>`
- Replace `String` with `heapless::String<N>`
- Replace `HashMap` with `heapless::FnvIndexMap<K, V, N>`
- Pre-allocate all memory at initialization (arena pattern acceptable)
- Use `tinyvec` or `arrayvec` for small, stack-based collections

**Cargo.toml Configuration:**
```toml
[dependencies]
heapless = "0.8"

[features]
default = []
std = []  # Enable only for testing
```

**Clippy Enforcement:**
```rust
#![deny(clippy::vec_init_then_push)]
#![deny(clippy::large_stack_arrays)]  // Configure threshold
```

---

### Rule 4: Function Length Limit — 60 Lines Maximum

**Original Intent:** Ensure functions are small enough to understand, test, and verify completely.

**Rust Adaptation:**

```rust
// ❌ FORBIDDEN: Long functions
fn process_everything(data: &mut Data) -> Result<Output, Error> {
    // 200 lines of code...
}

// ✅ REQUIRED: Decomposed functions ≤60 lines each
fn process_everything(data: &mut Data) -> Result<Output, Error> {
    validate_input(data)?;
    let intermediate = transform_data(data)?;
    let analyzed = analyze_results(&intermediate)?;
    generate_output(&analyzed)
}

fn validate_input(data: &Data) -> Result<(), Error> {
    // ≤60 lines
}

fn transform_data(data: &mut Data) -> Result<Intermediate, Error> {
    // ≤60 lines
}
```

**Rust-Specific Guidelines:**
- Maximum 60 lines per function (including signature, braces, comments)
- Each function should do one thing
- Use early returns with `?` operator to reduce nesting
- Extract match arms into separate functions if complex
- Module organization: group related small functions

**Clippy Enforcement:**
```toml
# clippy.toml
too-many-lines-threshold = 60
```

```rust
#![deny(clippy::too_many_lines)]
```

---

### Rule 5: Assertion Density — Minimum 2 Per Function

**Original Intent:** Defensive programming catches bugs early; assertions document invariants.

**Rust Adaptation:**

```rust
// ❌ INSUFFICIENT: No invariant checking
fn calculate_average(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

// ✅ REQUIRED: Assert preconditions and invariants
fn calculate_average(values: &[f64]) -> Result<f64, MathError> {
    // Precondition assertions
    debug_assert!(!values.is_empty(), "values must not be empty");
    debug_assert!(
        values.iter().all(|v| v.is_finite()),
        "all values must be finite"
    );

    if values.is_empty() {
        return Err(MathError::EmptyInput);
    }

    let sum: f64 = values.iter().sum();
    let avg = sum / values.len() as f64;

    // Postcondition assertion
    debug_assert!(avg.is_finite(), "average must be finite");

    Ok(avg)
}
```

**Rust-Specific Guidelines:**
- Use `debug_assert!` for invariants (zero runtime cost in release)
- Use `assert!` for critical invariants that must always be checked
- Document what each assertion verifies
- Assert preconditions at function entry
- Assert postconditions before returning
- Assert loop invariants within complex loops
- Use `static_assertions` crate for compile-time checks

**Types of Assertions:**
```rust
// Compile-time assertions
use static_assertions::const_assert;
const_assert!(BUFFER_SIZE >= MIN_REQUIRED_SIZE);
const_assert!(core::mem::size_of::<Packet>() <= MTU);

// Runtime preconditions
debug_assert!(index < self.len());
debug_assert!(ptr.is_aligned());

// Postconditions
debug_assert!(result >= 0.0 && result <= 1.0);

// Invariants
debug_assert!(self.is_valid());
```

---

### Rule 6: Minimize Variable Scope

**Original Intent:** Reduce state complexity and potential for misuse.

**Rust Adaptation:**

```rust
// ❌ FORBIDDEN: Variables declared far from use
fn process() -> Result<(), Error> {
    let temp_buffer: [u8; 256];  // Declared early
    let counter: usize;

    // ... 50 lines later ...

    temp_buffer = [0; 256];
    counter = 0;
}

// ✅ REQUIRED: Variables at smallest scope
fn process() -> Result<(), Error> {
    // Variables declared at point of use
    let validated = {
        let raw_input = read_input()?;
        validate(&raw_input)?  // raw_input drops here
    };

    for item in validated.iter() {
        let transformed = transform(item);  // Scoped to loop
        output(transformed)?;
    }

    Ok(())
}
```

**Rust-Specific Guidelines:**
- Declare variables at first use, not at function start
- Use blocks `{}` to limit scope explicitly
- Prefer immutable bindings (`let`) over mutable (`let mut`)
- Avoid `static mut` entirely — use atomics or interior mutability patterns
- Minimize lifetime parameters; prefer owned data where practical
- Use shadowing intentionally to narrow scope

**Clippy Enforcement:**
```rust
#![deny(clippy::let_underscore_must_use)]
#![deny(clippy::unused_self)]
```

---

### Rule 7: Check All Return Values and Validate Parameters

**Original Intent:** Never ignore errors; verify inputs at trust boundaries.

**Rust Adaptation:**

```rust
// ❌ FORBIDDEN: Ignoring Results
let _ = file.write_all(data);  // Error silently ignored
file.write_all(data).ok();     // Error discarded

// ❌ FORBIDDEN: Panicking on errors
let value = result.unwrap();   // Panic!
let value = result.expect(""); // Panic!

// ✅ REQUIRED: Explicit error handling
file.write_all(data)?;  // Propagate error

match result {
    Ok(value) => process(value),
    Err(e) => handle_error(e),
}

// ✅ REQUIRED: Parameter validation at boundaries
pub fn set_temperature(celsius: f64) -> Result<(), ConfigError> {
    // Validate at public API boundary
    if !(-273.15..=1_000_000.0).contains(&celsius) {
        return Err(ConfigError::TemperatureOutOfRange);
    }

    // Internal code can trust validated input
    internal_set_temp(celsius)
}
```

**Rust-Specific Guidelines:**
- **Never use `unwrap()` or `expect()` in production code**
- Use `?` operator to propagate errors
- Use `#[must_use]` attribute on functions with important return values
- Validate all inputs at trust boundaries (public APIs, FFI, I/O)
- Internal functions may trust already-validated data
- Use newtypes to encode validated state in the type system

**Clippy Enforcement:**
```rust
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::let_underscore_must_use)]
#![deny(clippy::result_unit_err)]
#![deny(unused_must_use)]
```

---

### Rule 8: Limit Macro Complexity

**Original Intent:** Preprocessor abuse creates unmaintainable, unanalyzable code.

**Rust Adaptation:**

```rust
// ❌ FORBIDDEN: Complex procedural macros in safety-critical code
#[derive(ComplexMacro)]  // Generates opaque code
struct Data { }

// ❌ FORBIDDEN: Macros that obscure control flow
my_macro! {
    if condition { return; }  // Hidden control flow
}

// ✅ ACCEPTABLE: Simple declarative macros
macro_rules! check_flag {
    ($flags:expr, $bit:expr) => {
        ($flags & (1 << $bit)) != 0
    };
}

// ✅ REQUIRED: Prefer functions over macros
#[inline]
const fn check_flag(flags: u32, bit: u32) -> bool {
    (flags & (1 << bit)) != 0
}

// ✅ ACCEPTABLE: Standard derive macros only
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Status {
    code: u8,
    flags: u16,
}
```

**Rust-Specific Guidelines:**
- Prefer `const fn` and `inline` functions over macros
- Only use derive macros from audited, stable crates
- Keep declarative macros simple and transparent
- Document what macros expand to
- Avoid procedural macros that generate complex logic
- Audit macro expansions with `cargo expand`

**Allowed Derive Macros (Audited):**
- `Debug`, `Clone`, `Copy`, `Default`
- `PartialEq`, `Eq`, `PartialOrd`, `Ord`, `Hash`
- `serde::Serialize`, `serde::Deserialize` (if audited)

---

### Rule 9: Restrict Unsafe Code

**Original Intent:** (C: Limit pointer arithmetic and function pointers)

**Rust Adaptation — This is Rust's most critical rule:**

```rust
// ❌ FORBIDDEN: Uncontrolled unsafe blocks
unsafe {
    let ptr = data.as_ptr();
    // Multiple unsafe operations
    *ptr.add(1) = value;
    call_c_function(ptr);
}

// ❌ FORBIDDEN: unsafe in application logic
fn business_logic(data: &mut Data) {
    unsafe { /* anything */ }
}

// ✅ REQUIRED: Isolated unsafe in audited modules
mod ffi_wrapper {
    //! # Safety
    //! This module wraps unsafe FFI calls to libfoo.
    //! All unsafe code has been audited for memory safety.

    /// # Safety
    /// - `ptr` must be valid and properly aligned
    /// - `ptr` must point to initialized memory
    /// - No other references to the memory may exist
    pub(crate) unsafe fn raw_read(ptr: *const u8) -> u8 {
        // SAFETY: Caller guarantees preconditions
        unsafe { *ptr }
    }
}

// ✅ REQUIRED: Safe wrappers around unsafe
pub fn safe_read(buffer: &[u8], index: usize) -> Option<u8> {
    buffer.get(index).copied()
}
```

**Rust-Specific Guidelines:**
- **Minimize unsafe to the absolute minimum**
- All `unsafe` blocks must have `// SAFETY:` comments
- All `unsafe fn` must document safety requirements in `# Safety` section
- Isolate unsafe into dedicated, audited modules
- Provide safe wrappers around all unsafe operations
- Run `cargo miri test` to verify unsafe code
- Use `#![forbid(unsafe_code)]` in application crates
- Only allow unsafe in dedicated `-sys` or `ffi` crates

**Clippy Enforcement:**
```rust
#![forbid(unsafe_code)]  // In application crates

// In FFI crates only:
#![deny(clippy::undocumented_unsafe_blocks)]
#![deny(clippy::multiple_unsafe_ops_per_block)]
```

---

### Rule 10: Maximum Compiler Strictness and Static Analysis

**Original Intent:** Catch issues at compile time; use every available tool.

**Rust Adaptation:**

```rust
// Required crate-level attributes
#![deny(warnings)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![deny(clippy::nursery)]
#![deny(clippy::cargo)]

// Safety-critical specific denials
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::unreachable)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![deny(clippy::dbg_macro)]
#![deny(clippy::print_stdout)]
#![deny(clippy::print_stderr)]

// Unsafe restrictions
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]

// Missing documentation
#![deny(missing_docs)]
#![deny(rustdoc::missing_crate_level_docs)]
```

**Required Toolchain:**
```bash
# Daily CI pipeline
cargo fmt --check                    # Formatting
cargo clippy -- -D warnings          # Lints
cargo test                           # Unit tests
cargo miri test                      # Undefined behavior check
cargo audit                          # Security vulnerabilities
cargo deny check                     # License & dependency audit
cargo +nightly udeps                 # Unused dependencies
```

**Cargo.toml Configuration:**
```toml
[profile.release]
overflow-checks = true  # Keep overflow checks in release
debug-assertions = false
lto = true
panic = "abort"

[lints.rust]
unsafe_code = "forbid"
missing_docs = "deny"

[lints.clippy]
all = "deny"
pedantic = "deny"
nursery = "warn"
unwrap_used = "deny"
expect_used = "deny"
```

---

## Summary: Rust's Built-in Advantages

Many Power of 10 rules are **automatically enforced** by Rust's compiler:

| Original Rule | Rust Enforcement |
|--------------|------------------|
| No `goto`, `setjmp`, `longjmp` | ✅ Language doesn't have these |
| Check return values | ✅ `#[must_use]`, `Result<T, E>` |
| Minimize scope | ✅ Ownership system, RAII |
| Pointer safety | ✅ Borrow checker, no null |
| Type safety | ✅ Strong static typing |
| Initialize before use | ✅ Compiler enforces |

**Rules requiring discipline in Rust:**
- No recursion (not compiler-enforced)
- Fixed loop bounds (requires design discipline)
- No heap allocation (requires `no_std` + discipline)
- Function length limits (requires Clippy)
- Assertion density (requires discipline)
- Unsafe minimization (requires governance)

---

## References

- [Original Power of 10 Paper](https://spinroot.com/gerard/pdf/P10.pdf) — Gerard Holzmann
- [Expanded Guidelines](https://spinroot.com/gerard/pdf/P10exp.pdf) — JPL Experience Report
- [Perforce: NASA's Rules](https://www.perforce.com/blog/kw/NASA-rules-for-developing-safety-critical-code)
- [Rust Safety-Critical Consortium](https://github.com/rustfoundation/safety-critical-rust-coding-guidelines)
- [Bringing Rust to Safety-Critical Space Systems](https://arxiv.org/html/2405.18135v1)
- [High Assurance Rust Book](https://highassurance.rs/chp1/why_this_book.html)
- [MISRA-Rust Mapping](https://github.com/PolySync/misra-rust/blob/master/MISRA-Rules.md)
- [Clippy Lints Reference](https://rust-lang.github.io/rust-clippy/master/index.html)
- [Embedded Rust Book](https://docs.rust-embedded.org/book/)
- [Heapless Collections](https://docs.rs/heapless)
