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
