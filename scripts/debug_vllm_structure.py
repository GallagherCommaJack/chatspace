#!/usr/bin/env python3
"""Debug script to inspect vLLM's internal structure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm import LLM

print("Loading vLLM model to inspect structure...")
llm = LLM(model="Qwen/Qwen2.5-3B", tensor_parallel_size=1, gpu_memory_utilization=0.3)

print("\n=== vLLM Engine Structure ===")
print(f"llm type: {type(llm)}")
print(f"llm attributes: {dir(llm)}")

print("\n=== Engine Attributes ===")
engine = llm.llm_engine
print(f"engine type: {type(engine)}")
print(f"engine attributes: {[a for a in dir(engine) if not a.startswith('_')]}")

print("\n=== Model Executor ===")
if hasattr(engine, 'model_executor'):
    executor = engine.model_executor
    print(f"executor type: {type(executor)}")
    print(f"executor attributes: {[a for a in dir(executor) if not a.startswith('_')]}")

    print("\n=== Looking for workers ===")
    if hasattr(executor, 'driver_worker'):
        print("✓ Found driver_worker")
        worker = executor.driver_worker
    elif hasattr(executor, 'workers'):
        print(f"✓ Found workers list (len={len(executor.workers)})")
        worker = executor.workers[0] if executor.workers else None
    else:
        print("✗ No driver_worker or workers found")
        worker = None

    if worker:
        print(f"\nWorker type: {type(worker)}")
        print(f"Worker attributes: {[a for a in dir(worker) if not a.startswith('_')]}")

        print("\n=== Looking for model ===")
        if hasattr(worker, 'model_runner'):
            print("✓ Found model_runner")
            runner = worker.model_runner
            print(f"Model runner type: {type(runner)}")
            print(f"Model runner attributes: {[a for a in dir(runner) if not a.startswith('_')]}")

            if hasattr(runner, 'model'):
                print("✓ Found model in runner")
                model = runner.model
                print(f"Model type: {type(model)}")
                print(f"Model attributes (first 20): {[a for a in dir(model) if not a.startswith('_')][:20]}")

                # Check for layers
                if hasattr(model, 'model'):
                    print("\n✓ Found model.model")
                    inner_model = model.model
                    print(f"Inner model type: {type(inner_model)}")
                    if hasattr(inner_model, 'layers'):
                        print(f"✓ Found layers! Length: {len(inner_model.layers)}")
                        print(f"Layer 0 type: {type(inner_model.layers[0])}")
                    else:
                        print("✗ No layers in model.model")
                elif hasattr(model, 'layers'):
                    print(f"\n✓ Found layers directly! Length: {len(model.layers)}")
                else:
                    print("\n✗ No layers found")
            else:
                print("✗ No model in model_runner")
        elif hasattr(worker, 'model'):
            print("✓ Found model directly in worker")
            model = worker.model
            print(f"Model type: {type(model)}")
        else:
            print("✗ No model_runner or model in worker")
else:
    print("✗ No model_executor in engine")

print("\n=== Checking for config ===")
try:
    # Try different paths to get config
    if hasattr(llm.llm_engine, 'model_config'):
        config = llm.llm_engine.model_config
        print(f"✓ Found model_config in engine")
        print(f"Config type: {type(config)}")
        if hasattr(config, 'hf_config'):
            print(f"Hidden size: {config.hf_config.hidden_size}")
    else:
        print("Trying to get config from model...")
        # Will populate above
except Exception as e:
    print(f"Error getting config: {e}")

print("\n=== Summary ===")
print("Use this information to update _get_base_model() in vllm_steer_model.py")
