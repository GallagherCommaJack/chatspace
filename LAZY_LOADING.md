# Lazy Loading Optimization

This document describes the lazy loading optimizations applied to keep CLI startup fast.

## Problem

Before optimization:
```bash
$ time chatspace --help
# Takes 4+ seconds due to importing torch, transformers, datasets, etc.
```

The issue was that the CLI imported the entire `hf_embed` package at module load time, which transitively imported:
- `torch` (~2-3 seconds)
- `sentence-transformers` (~1 second)
- `datasets` library
- `pyarrow`
- Other heavy dependencies

This made even simple commands like `chatspace --help` very slow.

## Solution

We applied **lazy loading** to defer heavy imports until they're actually needed.

### Changes Made

#### 1. cli.py - Defer hf_embed imports

**Before**:
```python
from .hf_embed import SentenceTransformerConfig, run_sentence_transformer

def handle_embed_hf(args):
    cfg = SentenceTransformerConfig(...)
    run_sentence_transformer(cfg)
```

**After**:
```python
# No top-level import

def handle_embed_hf(args):
    # Import only when actually running embed-hf command
    from .hf_embed import SentenceTransformerConfig, run_sentence_transformer

    cfg = SentenceTransformerConfig(...)
    run_sentence_transformer(cfg)
```

#### 2. hf_embed/__init__.py - Lazy module attribute

**Before**:
```python
from .config import SentenceTransformerConfig
from .pipeline import run_sentence_transformer  # Imports torch!
```

**After**:
```python
from typing import TYPE_CHECKING
from .config import SentenceTransformerConfig

if TYPE_CHECKING:
    from .pipeline import run_sentence_transformer

def __getattr__(name: str):
    """Lazy import to avoid loading heavy dependencies at import time."""
    if name == "run_sentence_transformer":
        from .pipeline import run_sentence_transformer
        return run_sentence_transformer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

This uses Python's `__getattr__` mechanism to load `run_sentence_transformer` only when accessed.

#### 3. pipeline.py - Defer torch import

**Before**:
```python
import torch

# Enable tf32 tensor cores
torch.set_float32_matmul_precision('high')

def run_sentence_transformer(cfg):
    # ...
```

**After**:
```python
# No module-level torch import

def _encoder_worker(...):
    # Import torch inside the worker process
    import torch
    torch.set_float32_matmul_precision('high')

    # Rest of encoder logic
```

Since the encoder runs in a separate process, torch is now only imported in that process, not in the main CLI process.

## Results

### Performance Comparison

| Command | Before | After | Improvement |
|---------|--------|-------|-------------|
| `chatspace --help` | ~4.2s | ~0.08s | **52x faster** |
| `chatspace embed-hf --help` | ~4.2s | ~0.08s | **52x faster** |
| Config import only | N/A | ~0.07s | Fast |
| Full pipeline import | ~4.2s | ~4.2s | Same (only when needed) |

### Timing Breakdown

```bash
# Fast: No heavy imports
$ time chatspace --help
real    0m0.082s

# Still fast: Only imports config (pure Python dataclass)
$ time python -c "from chatspace.hf_embed import SentenceTransformerConfig"
real    0m0.070s

# Slow: Actually imports torch and friends (but only when needed)
$ time python -c "from chatspace.hf_embed import run_sentence_transformer"
real    0m4.170s
```

## Benefits

1. **Fast CLI responsiveness**: Help, version, and argument parsing are instant
2. **Better UX**: Users don't wait 4+ seconds just to see `--help`
3. **Efficient development**: Testing CLI argument parsing is fast
4. **No runtime overhead**: Heavy imports still happen, just at the right time
5. **Multiprocessing synergy**: Torch is loaded in encoder process, not main process

## Implementation Details

### Python's __getattr__ for Modules

We use `__getattr__` (PEP 562) to intercept attribute access on the module:

```python
def __getattr__(name: str):
    if name == "run_sentence_transformer":
        from .pipeline import run_sentence_transformer  # Import on first access
        return run_sentence_transformer
    raise AttributeError(...)
```

This is called when someone does:
```python
from chatspace.hf_embed import run_sentence_transformer  # Triggers __getattr__
```

But NOT when:
```python
from chatspace.hf_embed import SentenceTransformerConfig  # Direct import, fast
```

### TYPE_CHECKING Guard

For type checkers and IDEs, we use `TYPE_CHECKING`:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import run_sentence_transformer
```

This gives IDEs full type information without actually importing at runtime.

### Per-Process Imports

Since we use multiprocessing, torch is now imported **inside the encoder process**:

```
Main Process (CLI):
  - No torch import ✓
  - Fast startup ✓

Encoder Process:
  - Imports torch when needed ✓
  - GIL-free inference ✓
```

## Caveats

1. **First access is still slow**: The first time you actually run `embed-hf`, it will take 4+ seconds to load torch. This is unavoidable.

2. **Import errors are deferred**: If there's a typo or missing dependency, you won't see the error until you actually use the function.

3. **Type checking**: Must use `TYPE_CHECKING` guard for type checkers to work correctly.

## Verification

To verify lazy loading is working:

```bash
# Should be fast (~80ms)
time chatspace --help

# Should also be fast (~80ms)
time chatspace embed-hf --help

# Should be slow (~4s) because it actually needs torch
time chatspace embed-hf --dataset HuggingFaceTB/smoltalk --max-rows 1
```

## Related Patterns

This pattern is used by many Python projects:

- **Django**: Defers ORM imports until needed
- **Pandas**: Lazy loads optional dependencies
- **Click**: Defers command imports in groups
- **FastAPI**: Lazy loads Pydantic validators

## References

- [PEP 562 - Module __getattr__](https://peps.python.org/pep-0562/)
- [Python Import System](https://docs.python.org/3/reference/import.html)
- [Multiprocessing Migration](MULTIPROCESSING_MIGRATION.md) - Why torch imports are process-local