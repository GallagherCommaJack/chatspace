#!/usr/bin/env python3
"""Debug script to explore engine_core in vLLM V1."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm import LLM

print("Loading vLLM model...")
llm = LLM(model="Qwen/Qwen2.5-3B", tensor_parallel_size=1, gpu_memory_utilization=0.3)

print("\n=== Exploring engine_core ===")
engine = llm.llm_engine
if hasattr(engine, 'engine_core'):
    core = engine.engine_core
    print(f"engine_core type: {type(core)}")
    print(f"engine_core attributes: {[a for a in dir(core) if not a.startswith('_')]}")

    # Check if we can access workers through engine_core
    for attr in ['workers', 'worker', 'model_executor', 'gpu_worker']:
        if hasattr(core, attr):
            print(f"\n✓ Found {attr} in engine_core")
            obj = getattr(core, attr)
            print(f"  Type: {type(obj)}")
            if hasattr(obj, '__len__'):
                try:
                    print(f"  Length: {len(obj)}")
                except:
                    pass

# Try checking vLLM's internal structure
print("\n=== Checking for alternate model access ===")
print("Trying llm.llm_engine.__dict__.keys():")
print(list(engine.__dict__.keys()))

print("\n=== Recommendation ===")
print("V1 engine runs model in separate process - cannot directly install hooks")
print("Options:")
print("  1. Use vLLM 0.6.x (has V0 architecture with model_executor)")
print("  2. Use custom logit processors instead of hooks")
print("  3. Monkey-patch model loading in vLLM")
