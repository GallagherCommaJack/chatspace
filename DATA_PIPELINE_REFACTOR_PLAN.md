# Data Pipeline Refactor Plan

## Context
- Multiple persona-processing scripts under `scripts/` replicate dataset loading, filtering, and evaluation logic that also appears inside the `chatspace/steering/` package.
- Helpers for steering vector training and evaluation are duplicated across scripts and library modules, making maintenance difficult.
- Shared utilities (e.g., sanitising dataset names, MiniLM classifier training, hidden state extraction) exist in several slightly divergent copies.

## Objectives
- Unify dataset discovery, filtering, and Hugging Face materialisation behind a first-class library module (e.g., `chatspace.persona`) that scripts and steering code import.
- Expose a common evaluation toolkit that handles hidden-state extraction, projection-based classifiers, and MiniLM scoring for reuse across CLI workflows.
- Consolidate steering-training orchestration (token-budget sampling, vector reset, Trainer config) into `chatspace/steering` so batch scripts become thin wrappers.
- Centralise generic helpers (path sanitation, workspace roots, manifest writing) to remove copy/paste between job runners, rollouts, and training scripts.

## Proposed Workstreams
1. **Persona Dataset Module**
   - Create a module that standardises path conventions, dataset listing, HF/parquet loading, and score-based filtering.
   - Migrate `persona_to_hf.py`, steering data loaders, and persona dataset scripts to use the new module.

2. **Shared Evaluation Utilities**
   - Build a reusable hidden-state extraction + classifier/regressor layer within `chatspace/steering` (or a sibling `chatspace/eval` package).
   - Refactor `scripts/eval_steering_as_classifier.py`, `scripts/eval_comprehensive_classifiers.py`, and rollout scoring to import these utilities.

3. **Steering Training Core**
   - Factor vector reset, dataset token slicing, and Trainer setup helpers into `chatspace/steering`.
   - Update `scripts/train_all_steering.py` and `scripts/sweep_learning_rates.py` to call those helpers directly.

4. **Utility Consolidation**
   - Collect cross-cutting helpers (e.g., `_sanitize_component`, workspace path builders) into a shared utilities module.
   - Replace bespoke copies throughout `chatspace/steering/job.py`, rollout generation scripts, and related tooling.

## Notes
- Existing tests under `tests/` are acknowledged but not a primary focus for this refactor phase; they can be revisited once shared modules stabilise.
