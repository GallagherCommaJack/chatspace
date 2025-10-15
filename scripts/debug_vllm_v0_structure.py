#!/usr/bin/env python3
"""Debug script to inspect vLLM V0 engine structure (with disable_v1=True)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm import LLM

print("Loading vLLM model with V0 engine (disable_v1=True)...")
try:
    # Force V0 engine
    llm = LLM(
        model="Qwen/Qwen2.5-3B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        disable_v1=True,  # Use V0 engine
    )

    print("\n=== vLLM V0 Engine Structure ===")
    engine = llm.llm_engine
    print(f"Engine type: {type(engine)}")
    print(f"Engine has model_executor: {hasattr(engine, 'model_executor')}")

    if hasattr(engine, 'model_executor'):
        executor = engine.model_executor
        print(f"\nExecutor type: {type(executor)}")

        # Try to get worker
        worker = None
        if hasattr(executor, 'driver_worker'):
            worker = executor.driver_worker
            print("✓ Found driver_worker")
        elif hasattr(executor, 'workers') and executor.workers:
            worker = executor.workers[0]
            print(f"✓ Found workers[0]")

        if worker:
            print(f"Worker type: {type(worker)}")

            # Try to get model
            if hasattr(worker, 'model_runner') and hasattr(worker.model_runner, 'model'):
                model = worker.model_runner.model
                print(f"✓ Found model via model_runner")
                print(f"Model type: {type(model)}")

                # Check for layers
                if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                    print(f"✓ Found model.model.layers (length: {len(model.model.layers)})")
                    print(f"Layer type: {type(model.model.layers[0])}")
                    print("\n✅ SUCCESS: Can access layers for hooks!")
                else:
                    print("✗ Cannot find layers")
            else:
                print("✗ Cannot find model")

    print("\n✓ V0 engine initialization successful")

except Exception as e:
    print(f"\n✗ Error with V0 engine: {e}")
    import traceback
    traceback.print_exc()
