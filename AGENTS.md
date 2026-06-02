# Repository Guidelines

## Project Structure

Core Python source lives in `mast3r_fusion/`. The main runtime entrypoint is
`main.py`, with offline or batch-oriented logic in `main_loop.py`. Frontend model
abstractions are under `mast3r_fusion/frontend_model/`; MASt3R utilities live in
`mast3r_fusion/mast3r_utils.py`, and PI3X helpers live in
`mast3r_fusion/pi3x_utils.py`.

Dataset and camera configs are in `config/`. Evaluation scripts are in
`evaluation/`. Shader and visualization assets are in `resources/`. Operational
notes are kept in `docs/`.

## Current Task Scope

Local work on this machine is limited to algorithm development, local static
checks, and documentation updates directly related to the implementation.

Complete runtime validation, server environment adaptation, production-data
debugging, and post-deployment effect verification should be done on the server
environment after the branch is synchronized.

## Common Commands

Install editable dependencies:

```bash
pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install --no-build-isolation -e .
```

Run the SLAM pipeline:

```bash
python main.py --config config/base_kitti360.yaml --calib config/intrinsics_kitti360.yaml --dataset <path>
```

Select a frontend explicitly:

```bash
python main.py --frontend-model mast3r ...
python main.py --frontend-model pi3x --frontend-weights checkpoints/pi3x/model.safetensors ...
```

Lightweight syntax check:

```bash
python -m compileall main.py mast3r_fusion evaluation
```

Direct frontend output smoke test:

```bash
python tools/test_frontend_output.py --frontend-model mast3r --config config/base_kitti360.yaml --output-dir frontend_test_outputs/mast3r
python tools/test_frontend_output.py --frontend-model pi3x --frontend-weights checkpoints/pi3x/model.safetensors --config config/base_kitti360.yaml --output-dir frontend_test_outputs/pi3x
```

The smoke-test tool saves input images, confidence/depth maps, and `.ply` point
clouds for manual inspection.

## Coding Style

Use Python 3 style with 4-space indentation. Keep changes local to the relevant
module and follow existing naming: `snake_case` for functions and variables,
`PascalCase` for classes, and short adapter names such as `MASt3RAdapter` and
`PI3Adapter`.

Prefer explicit tensor shape comments where model interfaces are non-obvious.
Avoid broad refactors in evaluation scripts unless requested.

## Testing Guidelines

There is no formal test suite in this repository. Use `compileall` for syntax
validation, then run a short dataset range before full evaluation:

```bash
python main.py --start_from 0 --end_at 20 --no-viz ...
```

For frontend changes, first verify MASt3R still runs, then test PI3X separately.
Check generated result files and watch for low match fractions, Cholesky
failures, missing checkpoints, or import errors.

## Commit Guidelines

Recent commit messages are short, imperative summaries, for example
`Add PI3X frontend adapter` or `fix small imu syntax bug`.

Keep each commit focused. Do not mix local notes, evaluation plot cleanup,
runtime adapter changes, and unrelated workspace changes unless the user asks for
that exact grouping.

Before server testing, run:

```bash
git status --short
```

Do not commit datasets, checkpoints, logs, generated result files, or local
frontend test outputs.
