# Repository Guidelines

## Project Structure & Module Organization
`mbrl/` contains the core model-based RL code (controllers, models, environments, logging, and entrypoint logic).  
`experiments/cee_us/settings/` stores YAML experiment recipes by environment and phase (`curious_exploration/`, `zero_shot_generalization/`).  
`experiments/cee_us/hooks/` contains injectable hook functions used by `mbrl/main.py`.  
`datasets/` holds rollout datasets for analysis; `docs/` and `zh_docs/` contain project documentation.  
Use `scripts/` for environment/bootstrap helpers (for example MuJoCo setup). Large/generated artifacts in `results/`, `downloads/`, and `videos/` are ignored by Git.

## Build, Test, and Development Commands
- `python3.8 -m venv .venv && source .venv/bin/activate`: create and activate local environment.
- `pip install -r requirements.txt`: install full runtime dependencies (includes `mujoco-py`).
- `pip install -r requirements.no_mujoco.txt`: install analysis/dev dependencies without MuJoCo.
- `pip install -e .`: editable install for local development.
- `pre-commit install`: enable commit-time checks.
- `pre-commit run --all-files`: run formatting/lint checks across the repository.
- `python mbrl/main.py experiments/cee_us/settings/construction/curious_exploration/gnn_ensemble_cee_us.yaml`: run a representative training job.

## Coding Style & Naming Conventions
Target Python 3.8, 4-space indentation, and 120-char line length (`pyproject.toml`, `.flake8`).  
Run `black`, `isort --profile black`, and `flake8` on `mbrl/` and `experiments/`.  
Use `snake_case` for modules/functions/variables and `PascalCase` for classes.  
Name configs descriptively (for example `gnn_ensemble_cee_us_zero_shot_stack.yaml`) and keep hook modules focused (one hook per file where practical).

## Testing Guidelines
There is currently no dedicated `tests/` suite or enforced coverage gate.  
Minimum validation for changes:
1. `pre-commit run --all-files`
2. Smoke-run at least one relevant YAML config through `mbrl/main.py`
3. Verify expected logs/metrics in the configured working directory (and TensorBoard when applicable)

## Commit & Pull Request Guidelines
Recent history mostly follows a Conventional Commit style (for example `docs(zh): ...`).  
Prefer `<type>(<scope>): <imperative summary>` (`feat`, `fix`, `refactor`, `docs`, `chore`).  
PRs should include: purpose, key files/configs changed, exact reproduction command(s), environment assumptions (Python/MuJoCo/CUDA), and before/after metrics or screenshots for behavior changes.
