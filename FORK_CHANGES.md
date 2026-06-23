# Fork changes — `openai-only` branch

This fork of [Collab-Overcooked](https://github.com/YusaeMeow/Collab-Overcooked)
(base commit `f7b66f0`, MIT) is trimmed to a single purpose: run episodes against
**any OpenAI-compatible HTTP endpoint**, on the **latest Python**, with a minimal
dependency set and no package mirrors. The grid/MDP, the macro→primitive compiler,
the recipes, and the Referential Action Trajectories are unchanged.

## Model layer — `src/collab/modules.py`
- Replaced the per-model-name `if/elif` dispatch in `Module.query()` with **one**
  OpenAI client built from the environment (`OPENAI_BASE_URL`, `OPENAI_API_KEY`,
  model passed through as-is; `MODEL_MAX_TOKENS` optional).
- Removed the hardcoded `TOKEN_LIMIT_TABLE`; `restrict_dialogue()` now reads a
  single `MODEL_CONTEXT_LIMIT` (env, default 8192).
- Removed the `model_dirname` path prefix on the model id.
- Token counting now uses **`tiktoken`** only (cl100k_base fallback for non-OpenAI
  ids), dropping `transformers.AutoTokenizer` and the bundled `lib/llama_tokenizer`.
- `parse_response()` simplified to the standard SDK shape.
- Example retrieval (`get_top_k_similar_example`, `load_embedding`) disabled and the
  embedding-based `if_two_sentence_similar_meaning` replaced with a local
  `difflib` ratio — removes the OpenAI embeddings call (unsupported by many local
  servers) and the `pandas` / `scipy.spatial` usage.
- API key read from `$OPENAI_API_KEY` (falls back to `openai_key.txt` if present).

## Prompt directory override — `src/main.py`, `src/collab/collab.py`
- `PROMPT_DIR` now reads `$COLLAB_RECON_PROMPT_DIR` first, falling back to the
  built-in `cwd/prompts`. This lets the harness point an episode at a run-local
  prompts directory (with per-configuration `gpt/{chef,assistant}_skill.txt`
  rendered from the experiment config) without mutating the tracked prompt files.
  Mirrors the env-based model routing seam; behaviour is unchanged when the
  variable is unset.

## Human/web interface removed
- Deleted `src/collab/web_util.py` and `src/service.py`; removed the `human`
  branches and the `web_util` import from `src/main.py`. Drops `websockets`,
  `websocket-client`, `aiohttp`.

## Latest-Python modernisation
- `src/main.py`, `src/evaluation.py`: replaced `from distutils.util import strtobool`
  with a local helper (distutils is removed in Python 3.12+).
- `gym` → **`gymnasium`** (the ~4 sites: `register`, `gym.Env`,
  `gym.spaces.Discrete/Box`) in `overcooked_ai_py/__init__.py`,
  `mdp/overcooked_env.py`, and `setup.py`.

## Dependencies / mirrors
- New minimal `requirements.txt` (openai, numpy≥2.1, scipy, gymnasium, tiktoken,
  rich, tqdm, importlib-metadata, setuptools). Dropped: torch, fairscale, mpi4py,
  transformers, sentencepiece, tokenizers, dtw-python, scikit-learn, plotly,
  matplotlib, seaborn, pygame, blobfile, fire, protobuf, websockets/aiohttp,
  networkx, sympy, pandas.
- Deleted `lib/overcooked_ai/environment.yml` (Tsinghua mirrors, conda) and the
  broken root `dockerfile`. No conda, no mirrors anywhere.

## Bug fixes — `src/collab/collab.py` (behaviour-faithful)

Pre-existing defects in the plan->action parsing, fixed without changing the
intended semantics (the no-communication path is unchanged except for the dropped
phantom action):

- `generate_ml_action`: `ml_action == "wait(1)"` was a `==` comparison (a no-op)
  instead of an assignment `=` in the "no plan, no communication" fallback. The
  downstream `if ml_action == "": ...` re-parse masked it, so the outcome was
  unchanged, but the statement was dead.
- `parse_ml_action_top`: the follow-on action queue was built from the *unfiltered*
  split list with `index > 0`. When a `request/accept/deny/clarify` segment preceded
  the chosen action, that action sits at index > 0 and was **re-queued**, so it
  executed twice (once as `ml_action`, once from the queue). The queue now starts
  after the chosen action's index. Empty segments (the trailing `""` from
  `plan.split(";")` on a plan ending in `;`) are also skipped, removing a phantom
  wait step from the queue.
