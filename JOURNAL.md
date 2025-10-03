# Engineering Journal

## 2025-10-03
- Updated `chatspace/steering/train.py` to expose knob for gradient checkpointing, device mapping, epochs, and to print dataset/token counts before training. Trainer now skips Hugging Face model card writes and persists only the learnable steering vector plus config via `QwenSteerModel.save_pretrained`.
- Added lightweight serialization helpers to `chatspace/steering/model.py` (`save_pretrained`/`from_pretrained`) so checkpoints store just `steering_vector.pt` and `steering_config.json` instead of the full 32B model weights.
- Gradient-checkpointed runs currently fail during trainer initialization: Accelerate keeps layers on the `meta` device (`Cannot copy out of meta tensor`). We disable `low_cpu_mem_usage` when using `device_map=auto`, but need a clean GPU to confirm.
- Recent full-trait runs stalled while writing safetensors checkpoints (GPU utilization dropped to ~0%, disk filled with 62 GB models). The new save path should prevent this; pending validation once training resumes.
- GPU state is degraded: `nvidia-smi` reports ~85 GB VRAM in use with no processes, likely a leaked driver allocation after earlier OOM attempts. `nvidia-smi --gpu-reset` is unsupported; expect to power-cycle/reboot before rerunning the 32B model.
- Next steps: (1) smoke-test pipeline with a smaller base model (`Qwen/Qwen3-0.6B`) while GPU is unstable, (2) after reset, rerun the 32B trait training for one epoch and verify the compact checkpoint produces usable steering vectors, (3) add reload validation and extend to additional traits once stable.

## 2025-10-03 (Later)
- Smoke-tested the steering pipeline with `Qwen/Qwen3-0.6B` on the analytical trait: 1 epoch (~100k tokens) completed in ~2.4s with gradient checkpointing enabled, confirming the CLI changes and lightweight checkpoint path. Run artifacts (`steering_vector.pt`, `steering_config.json`) stored under `/workspace/steering_runs/qwen3-0.6b_analytical_epoch1` and reload successfully via `QwenSteerModel.from_pretrained`.
