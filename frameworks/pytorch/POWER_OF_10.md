# The Power of 10 Rules for Safety-Critical PyTorch Code

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

## The Power of 10 Rules — PyTorch Edition

### Rule 1: Simple Control Flow — No Recursive Networks, Guard Clauses

**Original Intent:** Eliminate complex control flow that impedes static analysis.

**PyTorch Adaptation:**

```python
# BAD: Recursive model forward
class RecursiveRNN(nn.Module):
    def forward(self, x, hidden, depth=10):
        if depth == 0:
            return hidden
        hidden = self.cell(x, hidden)
        return self.forward(x, hidden, depth - 1)  # Stack overflow risk

# BAD: Deeply nested training logic
def train_step(model, batch, optimizer):
    if batch is not None:
        if batch[0].shape[0] > 0:
            if model.training:
                outputs = model(batch[0])
                if outputs is not None:
                    # Finally compute loss...

# GOOD: Iterative implementation
class IterativeRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            nn.RNNCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

    def forward(self, x: Tensor, hidden: Tensor) -> Tensor:
        for cell in self.cells:
            hidden = cell(x, hidden)
        return hidden

# GOOD: Guard clauses with early returns
def train_step(
    model: nn.Module,
    batch: tuple[Tensor, Tensor],
    optimizer: Optimizer,
) -> float:
    if batch[0].shape[0] == 0:
        return 0.0

    assert model.training, "Model must be in training mode"

    optimizer.zero_grad()
    outputs = model(batch[0])
    loss = F.cross_entropy(outputs, batch[1])
    loss.backward()
    optimizer.step()

    return loss.item()
```

**Guidelines:**
- No recursive `forward()` methods
- Use `nn.ModuleList` for repeated layers
- Maximum 3-4 levels of nesting
- Use guard clauses for input validation
- Prefer iterative over recursive algorithms

---

### Rule 2: Bounded Loops — Fixed Epochs, Batch Limits

**Original Intent:** Ensure all loops terminate and can be analyzed statically.

**PyTorch Adaptation:**

```python
# BAD: Unbounded training loop
def train(model, dataloader):
    while True:  # No termination condition
        for batch in dataloader:
            loss = train_step(model, batch)
            if loss < 0.001:
                return  # Might never trigger

# BAD: Unbounded data loading
def load_all_data(path: str) -> list[Tensor]:
    data = []
    for file in Path(path).glob("*.pt"):
        data.append(torch.load(file))  # Unbounded memory
    return data

# GOOD: Bounded training with explicit limits
MAX_EPOCHS = 1000
MAX_STEPS_PER_EPOCH = 10_000

def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    num_epochs: int,
) -> dict[str, list[float]]:
    assert 0 < num_epochs <= MAX_EPOCHS, f"Epochs must be 1-{MAX_EPOCHS}"

    history = {"loss": [], "accuracy": []}

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        steps = 0

        for batch in dataloader:
            if steps >= MAX_STEPS_PER_EPOCH:
                break

            loss = train_step(model, batch, optimizer)
            epoch_loss += loss
            steps += 1

        history["loss"].append(epoch_loss / max(steps, 1))

    return history

# GOOD: Bounded data iteration
MAX_SAMPLES = 100_000

def load_dataset(path: str, max_samples: int = MAX_SAMPLES) -> Dataset:
    dataset = CustomDataset(path)
    if len(dataset) > max_samples:
        indices = torch.randperm(len(dataset))[:max_samples]
        dataset = Subset(dataset, indices.tolist())
    return dataset
```

**Guidelines:**
- Define `MAX_EPOCHS` and `MAX_STEPS` constants
- Use `range(n)` instead of `while True`
- Limit dataset sizes with `Subset`
- Add early stopping with patience limits
- Assert bounds on hyperparameters

---

### Rule 3: Controlled Memory — Explicit GPU Management

**Original Intent:** Prevent unbounded memory growth.

**PyTorch Adaptation:**

```python
# BAD: Memory leak in training loop
def train(model, dataloader):
    total_loss = 0
    for batch in dataloader:
        loss = model(batch).mean()
        total_loss += loss  # Accumulates computation graph!

# BAD: Not clearing gradients
def train_step(model, batch, optimizer):
    outputs = model(batch[0])
    loss = F.cross_entropy(outputs, batch[1])
    loss.backward()
    optimizer.step()  # Gradients accumulate forever!

# BAD: Unnecessary GPU-CPU transfers
def evaluate(model, dataloader):
    predictions = []
    for batch in dataloader:
        pred = model(batch[0])
        predictions.append(pred.cpu().numpy())  # Transfer every batch
    return np.concatenate(predictions)

# GOOD: Proper memory management
def train_step(
    model: nn.Module,
    batch: tuple[Tensor, Tensor],
    optimizer: Optimizer,
) -> float:
    optimizer.zero_grad()  # Clear gradients first

    outputs = model(batch[0])
    loss = F.cross_entropy(outputs, batch[1])
    loss.backward()
    optimizer.step()

    return loss.item()  # .item() detaches from graph

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: Optimizer) -> float:
    total_loss = 0.0  # Python float, not tensor
    num_batches = 0

    for batch in dataloader:
        loss = train_step(model, batch, optimizer)
        total_loss += loss  # Adding Python float
        num_batches += 1

    return total_loss / max(num_batches, 1)

# GOOD: Efficient evaluation with no_grad
@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader) -> Tensor:
    model.eval()
    predictions = []

    for batch in dataloader:
        pred = model(batch[0])
        predictions.append(pred)

    return torch.cat(predictions)  # Single concatenation

# GOOD: Explicit memory cleanup
def train_with_cleanup(model: nn.Module, dataloader: DataLoader) -> None:
    try:
        for epoch in range(NUM_EPOCHS):
            train_epoch(model, dataloader)
    finally:
        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()
        gc.collect()
```

**Guidelines:**
- Always call `optimizer.zero_grad()` before backward
- Use `.item()` to extract scalar values (detaches from graph)
- Use `@torch.no_grad()` for evaluation
- Avoid accumulating tensors in lists during training
- Call `torch.cuda.empty_cache()` when switching tasks
- Use `del` to remove references before clearing cache

---

### Rule 4: Short Functions — Modular Model Architecture

**Original Intent:** Ensure functions are small enough to understand and verify.

**PyTorch Adaptation:**

```python
# BAD: Monolithic model
class GiantModel(nn.Module):
    def forward(self, x):
        # 200 lines of transformations...
        pass

# GOOD: Modular architecture with separate blocks
class ConvBlock(nn.Module):
    """Single convolutional block with norm and activation."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class Encoder(nn.Module):
    """Encoder with multiple conv blocks."""

    def __init__(self, channels: list[int]):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvBlock(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)
        ])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> list[Tensor]:
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features


class Model(nn.Module):
    """Complete model composing encoder and decoder."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.encoder = Encoder(config.encoder_channels)
        self.decoder = Decoder(config.decoder_channels)
        self.head = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        features = self.encoder(x)
        decoded = self.decoder(features)
        return self.head(decoded)

# GOOD: Separate training logic into functions
def compute_loss(
    outputs: Tensor,
    targets: Tensor,
    class_weights: Tensor | None = None,
) -> Tensor:
    """Compute weighted cross-entropy loss."""
    return F.cross_entropy(outputs, targets, weight=class_weights)


def compute_metrics(outputs: Tensor, targets: Tensor) -> dict[str, float]:
    """Compute accuracy and F1 score."""
    preds = outputs.argmax(dim=1)
    accuracy = (preds == targets).float().mean().item()
    return {"accuracy": accuracy}
```

**Guidelines:**
- Maximum 60 lines per `forward()` method
- Create separate `nn.Module` classes for reusable blocks
- Extract loss computation into separate functions
- Use config dataclasses for hyperparameters
- One model file per architecture

---

### Rule 5: Assertions — Validate Shapes and Types

**Original Intent:** Defensive programming catches bugs early.

**PyTorch Adaptation:**

```python
from typing import assert_type

# GOOD: Assert tensor shapes
class Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        batch_size, seq_len, embed_dim = q.shape

        # Preconditions
        assert q.shape == k.shape == v.shape, "Q, K, V must have same shape"
        assert embed_dim == self.embed_dim, f"Expected {self.embed_dim}, got {embed_dim}"
        assert q.dtype == torch.float32, f"Expected float32, got {q.dtype}"

        # Computation
        attn_output = self._compute_attention(q, k, v)

        # Postcondition
        assert attn_output.shape == (batch_size, seq_len, self.embed_dim)

        return attn_output

# GOOD: Validate data loader outputs
def validate_batch(batch: tuple[Tensor, Tensor]) -> None:
    """Validate batch format before training."""
    inputs, targets = batch

    assert inputs.dim() == 4, f"Expected 4D input, got {inputs.dim()}D"
    assert targets.dim() == 1, f"Expected 1D targets, got {targets.dim()}D"
    assert inputs.shape[0] == targets.shape[0], "Batch size mismatch"
    assert not torch.isnan(inputs).any(), "NaN in inputs"
    assert not torch.isinf(inputs).any(), "Inf in inputs"

def train_step(model: nn.Module, batch: tuple[Tensor, Tensor], optimizer: Optimizer) -> float:
    validate_batch(batch)

    optimizer.zero_grad()
    outputs = model(batch[0])

    # Assert output validity
    assert not torch.isnan(outputs).any(), "NaN in model outputs"

    loss = F.cross_entropy(outputs, batch[1])

    assert not torch.isnan(loss), "NaN loss detected"
    assert loss.item() < 1e6, f"Loss exploded: {loss.item()}"

    loss.backward()
    optimizer.step()

    return loss.item()

# GOOD: Validate checkpoint integrity
def load_checkpoint(path: str, model: nn.Module) -> dict:
    checkpoint = torch.load(path, weights_only=True)

    assert "model_state_dict" in checkpoint, "Missing model state"
    assert "optimizer_state_dict" in checkpoint, "Missing optimizer state"
    assert "epoch" in checkpoint, "Missing epoch"

    model.load_state_dict(checkpoint["model_state_dict"])

    return checkpoint
```

**Guidelines:**
- Assert tensor shapes at module boundaries
- Check for NaN/Inf in inputs and outputs
- Validate batch dimensions before training
- Assert hyperparameter constraints in `__init__`
- Validate checkpoint contents when loading

---

### Rule 6: Minimal Scope — Encapsulate State, Avoid Globals

**Original Intent:** Reduce state complexity and potential for misuse.

**PyTorch Adaptation:**

```python
# BAD: Global model and optimizer
model = None
optimizer = None

def train():
    global model, optimizer
    model = Model()
    optimizer = Adam(model.parameters())
    # ...

# BAD: Mutable default arguments
def create_model(layers: list = [64, 128, 256]):  # Shared mutable default!
    return Model(layers)

# GOOD: Encapsulate in Trainer class
@dataclass
class TrainerConfig:
    learning_rate: float = 1e-3
    num_epochs: int = 100
    batch_size: int = 32
    device: str = "cuda"


class Trainer:
    def __init__(self, model: nn.Module, config: TrainerConfig):
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, config.num_epochs)

        # Private state
        self._current_epoch = 0
        self._best_loss = float("inf")

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.config.num_epochs):
            self._current_epoch = epoch
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            self.scheduler.step()

        return history

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        # ...

    def _validate(self, loader: DataLoader) -> float:
        self.model.eval()
        # ...

# GOOD: Immutable defaults
def create_model(layers: tuple[int, ...] = (64, 128, 256)) -> Model:
    return Model(list(layers))

# GOOD: Context manager for device placement
@contextmanager
def device_context(device: str):
    """Ensure tensors are created on correct device."""
    with torch.device(device):
        yield
```

**Guidelines:**
- No global model or optimizer state
- Use dataclasses for configuration
- Encapsulate training state in a Trainer class
- Use immutable defaults (tuples, not lists)
- Keep device placement explicit

---

### Rule 7: Check Returns — Handle Training Failures

**Original Intent:** Never ignore errors; verify at trust boundaries.

**PyTorch Adaptation:**

```python
# BAD: Ignoring NaN losses
def train_step(model, batch, optimizer):
    loss = compute_loss(model, batch)
    loss.backward()  # NaN propagates silently
    optimizer.step()

# BAD: Unchecked model loading
def load_model(path):
    model = Model()
    model.load_state_dict(torch.load(path))  # May fail silently
    return model

# GOOD: Comprehensive error handling
class TrainingError(Exception):
    """Base exception for training failures."""
    pass

class NaNLossError(TrainingError):
    """Raised when loss becomes NaN."""
    pass

class GradientExplosionError(TrainingError):
    """Raised when gradients explode."""
    pass

def train_step(
    model: nn.Module,
    batch: tuple[Tensor, Tensor],
    optimizer: Optimizer,
    max_grad_norm: float = 1.0,
) -> float:
    optimizer.zero_grad()

    outputs = model(batch[0])
    loss = F.cross_entropy(outputs, batch[1])

    # Check for training failures
    if torch.isnan(loss):
        raise NaNLossError(f"NaN loss at step")

    if loss.item() > 1e6:
        raise GradientExplosionError(f"Loss exploded: {loss.item()}")

    loss.backward()

    # Check gradient norms
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if torch.isnan(total_norm):
        raise NaNLossError("NaN gradients detected")

    optimizer.step()

    return loss.item()

# GOOD: Safe model loading with validation
def load_model(
    path: str,
    model_class: type[nn.Module],
    config: ModelConfig,
    device: str = "cpu",
) -> nn.Module:
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Use weights_only=True for security (prevents arbitrary code execution)
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    if "model_state_dict" not in checkpoint:
        raise ValueError("Invalid checkpoint format")

    model = model_class(config)
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )

    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    return model.to(device)
```

**Guidelines:**
- Always check for NaN/Inf in loss
- Use gradient clipping and check gradient norms
- Use `weights_only=True` when loading checkpoints
- Handle missing/unexpected keys in state dict
- Define custom exceptions for training failures

---

### Rule 8: Limit Metaprogramming — Explicit Model Definition

**Original Intent:** Avoid constructs that create unanalyzable code.

**PyTorch Adaptation:**

```python
# BAD: Dynamic model construction with exec
def create_model_dynamic(config: dict):
    code = f"""
class DynamicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear({config['in']}, {config['out']})
    """
    exec(code)
    return DynamicModel()

# BAD: Overly clever metaprogramming
def auto_register_layers(module, layer_specs):
    for i, spec in enumerate(layer_specs):
        setattr(module, f"layer_{i}", eval(f"nn.{spec['type']}(**spec['params'])"))

# GOOD: Explicit model factory
def create_model(config: ModelConfig) -> nn.Module:
    """Factory function with explicit model creation."""
    if config.architecture == "resnet":
        return ResNet(
            num_classes=config.num_classes,
            depth=config.depth,
        )
    elif config.architecture == "transformer":
        return Transformer(
            num_classes=config.num_classes,
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
        )
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")

# GOOD: Explicit layer construction
class FlexibleModel(nn.Module):
    def __init__(self, layer_sizes: list[int], activation: str = "relu"):
        super().__init__()

        # Explicit activation mapping
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")

        act_class = activations[activation]

        # Explicit layer construction
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(act_class())

        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

# GOOD: Use hooks sparingly and document them
class FeatureExtractor:
    """Extract intermediate features using hooks."""

    def __init__(self, model: nn.Module, layer_name: str):
        self.features: Tensor | None = None
        self._hook = model.get_submodule(layer_name).register_forward_hook(
            self._hook_fn
        )

    def _hook_fn(self, module: nn.Module, input: Tensor, output: Tensor) -> None:
        self.features = output.detach()

    def remove(self) -> None:
        self._hook.remove()
```

**Guidelines:**
- Never use `exec()` or `eval()` for model construction
- Use explicit factory functions
- Map strings to classes via dictionaries
- Document any hooks clearly
- Prefer `nn.Sequential` and `nn.ModuleList` over dynamic attributes

---

### Rule 9: Type Safety — Type Hints for Tensors

**Original Intent:** (C: Restrict pointer usage for safety)

**PyTorch Adaptation:**

```python
from typing import TypeAlias, Literal
from torch import Tensor
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype

# Define type aliases for clarity
BatchSize: TypeAlias = int
SeqLen: TypeAlias = int
EmbedDim: TypeAlias = int

# GOOD: Type hints for all functions
def create_attention_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create causal attention mask."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=dtype), diagonal=1)
    return mask.masked_fill(mask == 1, float("-inf"))

# GOOD: Typed model with explicit tensor shapes in docstrings
class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim))
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input token IDs, shape (batch_size, seq_len)

        Returns:
            Output embeddings, shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len = x.shape
        x = self.embedding(x) + self.pos_encoding[:seq_len]

        for layer in self.layers:
            x = layer(x)

        return x

# GOOD: Using jaxtyping for runtime shape checking (optional)
@jaxtyped(typechecker=beartype)
def attention(
    q: Float[Tensor, "batch heads seq_q dim"],
    k: Float[Tensor, "batch heads seq_k dim"],
    v: Float[Tensor, "batch heads seq_k dim"],
) -> Float[Tensor, "batch heads seq_q dim"]:
    """Compute scaled dot-product attention with shape validation."""
    scale = q.shape[-1] ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, v)

# GOOD: Config with strict types
@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    batch_size: int
    num_epochs: int
    device: Literal["cpu", "cuda", "mps"]
    seed: int = 42

    def __post_init__(self) -> None:
        assert 0 < self.learning_rate < 1, "Learning rate must be in (0, 1)"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.num_epochs > 0, "Number of epochs must be positive"
```

**Guidelines:**
- Type hint all function signatures
- Document tensor shapes in docstrings
- Use `Literal` for constrained string parameters
- Use frozen dataclasses for configuration
- Consider `jaxtyping` for runtime shape checking
- Validate config in `__post_init__`

---

### Rule 10: Static Analysis — Reproducibility and Testing

**Original Intent:** Catch issues at development time.

**PyTorch Adaptation:**

```python
# Reproducibility setup
def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # For complete reproducibility (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

# Save complete experiment state
def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    config: TrainingConfig,
) -> None:
    """Save complete checkpoint for reproducibility."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": asdict(config),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }, path)
```

**pyproject.toml:**
```toml
[tool.mypy]
strict = true
plugins = ["numpy.typing.mypy_plugin"]
warn_return_any = true

[tool.ruff]
select = ["ALL"]
ignore = ["D203", "D213"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

```bash
# Required CI pipeline
ruff check src/                  # Linting
ruff format --check src/         # Formatting
mypy src/                        # Type checking
pytest tests/                    # Unit tests
pip-audit                        # Security vulnerabilities

# Model-specific checks
python -c "import torch; print(torch.__version__)"
python scripts/validate_model.py  # Shape tests
```

**Guidelines:**
- Set all random seeds for reproducibility
- Use `torch.use_deterministic_algorithms(True)` when needed
- Save complete RNG states in checkpoints
- Run mypy with strict mode
- Test model forward pass with various input shapes
- Use `weights_only=True` when loading untrusted checkpoints

---

## Summary: PyTorch Adaptation

| # | Original Rule | PyTorch Guideline |
|---|---------------|-------------------|
| 1 | No goto/recursion | No recursive forward, guard clauses |
| 2 | Fixed loop bounds | Fixed epochs, bounded data loading |
| 3 | No dynamic allocation | Explicit GPU memory, `no_grad()`, `.item()` |
| 4 | 60-line functions | Modular `nn.Module` blocks |
| 5 | 2+ assertions/function | Assert shapes, check NaN/Inf |
| 6 | Minimize scope | Trainer class, no global state |
| 7 | Check returns | Handle NaN loss, validate checkpoints |
| 8 | Limit preprocessor | Explicit factories, no `exec()` |
| 9 | Restrict pointers | Type hints, shape documentation |
| 10 | All warnings enabled | Reproducibility, mypy strict |

---

## References

- [Original Power of 10 Paper](https://spinroot.com/gerard/pdf/P10.pdf) — Gerard Holzmann
- [PyTorch Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [PyTorch Memory Management](https://pytorch.org/blog/understanding-gpu-memory-1/)
- [PyTorch Best Practices](https://www.learnpytorch.io/)
- [Clear GPU Memory Guide 2026](https://copyprogramming.com/howto/how-to-clear-gpu-memory-after-using-model)
