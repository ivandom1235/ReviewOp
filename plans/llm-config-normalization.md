# Plan: LLM Configuration Normalization & Dynamic Flow

> Source PRD: [User Request Conversation bc010664]

## Architectural decisions

Durable decisions for the LLM configuration flow:

- **Environment Variables**: Strip `REVIEWOP_` prefix. Use `{PROVIDER}_API_KEY` and `{PROVIDER}_MODEL` patterns.
- **Provider Resolution**: The `llm_provider` object will be initialized with provider-specific credentials, and the `llm_model_name` will be dynamically resolved if not explicitly provided.
- **Fallback Logic**: If a provider-specific model is missing, warn and fall back to a `DEFAULT_LLM_MODEL`.
- **Single Source of Truth**: Only the root `.env` will be utilized.

---

## Phase 1: Environment Normalization & .env Cleanup

**User stories**: 
- Simplify variable names (remove cringe prefixes)
- Consolidate to root .env

### What to build
1. Update `dataset_builder/code/llm_utils.py` and `build_dataset.py` to remove `REVIEWOP_` from all `os.environ.get`, `_optional_env`, and `_required_env` calls.
2. Update the root `.env` file:
    - `REVIEWOP_DEFAULT_LLM_PROVIDER` -> `DEFAULT_LLM_PROVIDER`
    - `REVIEWOP_LLM_TIMEOUT_SECONDS` -> `LLM_TIMEOUT_SECONDS`
    - Remove prefixes from all other variables.
3. Ensure no local `.env` files exist in subdirectories.

### Acceptance criteria
- [ ] No `REVIEWOP_` prefixes in `.env` for LLM settings.
- [ ] Code correctly pulls `DEFAULT_LLM_PROVIDER`.
- [ ] Only root `.env` exists in the repo.

---

## Phase 2: Dynamic Provider Flow in dataset_builder

**User stories**:
- Automatically get API key, link, and model from .env when provider is mentioned.
- Provider-specific model resolution with fallback and warnings.

### What to build
1. Implement a `resolve_provider_defaults(provider_name, user_model)` helper in `build_dataset.py` that:
    - Maps `openai` -> `OPENAI_MODEL`, `claude` -> `CLAUDE_MODEL`, etc.
    - If `user_model` is provided, use it.
    - If not, try the provider-specific env var.
    - If provider-specific env var is missing, warn and fall back to `DEFAULT_LLM_MODEL` (which must be defined in `.env`).
2. Update `main()` in `build_dataset.py` to use this helper before initializing `BuilderConfig`.
3. Update `AsyncLlmProvider` subclasses in `llm_utils.py` to ensure they use the standard names (e.g., `AsyncOpenAiProvider` should look for `OPENAI_API_KEY`).

### Acceptance criteria
- [ ] Running with `--llm-provider openai` automatically uses `gpt-4o-mini` (or whatever is in `OPENAI_MODEL`).
- [ ] Running with `--llm-provider claude` uses `CLAUDE_MODEL`.
- [ ] A clear warning is printed if a provider is specified but its specific model is missing from `.env`.

---

## Phase 3: Improved Connectivity Validation & UX

**User stories**:
- Provider-aware smoke test warnings.

### What to build
1. Update the "RunPod Connectivity Smoke Test" in `build_dataset.py` to:
    - Be renamed to "LLM Provider Connectivity Test".
    - Correctly display the active provider name in success/failure messages.
    - Provide specific advice (e.g., "Check your OPENAI_API_KEY") instead of generic RunPod advice.

### Acceptance criteria
- [ ] Failure with OpenAI shows "OpenAI connectivity probe failed" and mentions `OPENAI_API_KEY`.
- [ ] Success message shows correctly identified provider.
- [ ] `llm_provider` is only set to `None` if the connectivity test fails and we have no fallback.
