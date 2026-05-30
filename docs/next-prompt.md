Finalize Phase 4 of the llm_platform adapter refactoring.

  CONTEXT (this is a continuation — start by re-grounding):
  - Repo: /mnt/c/Programs/llm_platform. First read docs/technical_documentation.md (per CLAUDE.md),
    then read the persisted memory files under the project memory dir — especially
    "refactoring-roadmap.md" and "llm-platform-architecture-notes.md" — which record what
    Phases 0–3 and the Phase 4 subset already did and what was deferred.
  
  WHAT'S ALREADY DONE (do not redo):
  - Phase 2: shared AdapterBase helpers (_merge_additional_parameters, _build_usage,
    _callable_to_json_schema, _image_data_url/_document_xml, PDF + MAX_TOOL_ROUNDS consts);
    request_llm_with_functions is a concrete default that raises a uniform error.
  - Phase 3: OpenAICompatibleAdapter base (DeepSeek/OpenRouter are thin subclasses);
    ModelConfig cached + name-indexed; Mistral tool args via **kwargs; Google while-True bounded.
  - Phase 4 (partial): Conversation.last_assistant_id (collapsed the two vendor-named props);
    SSHCommandTool base + lazy xai_sdk import; files.py PDF parse-failure logging;
    ParameterNormalizer extracted from APIHandler + lazy adapter registration (ADAPTER_IMPORT_PATHS).
  - A capability flag `adaptive_thinking` was added to models_config.yaml + read by AnthropicAdapter
    (fixed an Opus 4.8 thinking 400). This is the START of the capability-model item below — extend
    the same pattern to the remaining inline gates.

  DEFERRED PHASE 4 ITEMS TO FINALIZE (these were left because they're big, behavior-sensitive,
  and need live-provider testing):
  1. Capability model in models_config.yaml — replace remaining model-name-literal gating with
     per-model flags, following the `adaptive_thinking` precedent: Mistral OCR/transcription routing
     (model == "mistral-ocr-latest" / "voxtral-mini-latest"), Anthropic streaming threshold
     (MAX_TOKENS_STREAMING_THRESHOLD), Google "gemini-3" substring checks.
  2. Lazy client construction — move provider SDK client creation out of adapter __init__ behind a
     lazily-created property; keep load_dotenv out of constructors. Also move the network fetch out of
     OpenAIAdapter._parse_response (container file retrieval) so parsing is pure/testable and the
     async path doesn't block.
  3. Full iterative tool-loop template — collapse the 5 recursive tool loops into one AdapterBase
     template method with provider hooks + a shared max-iterations guard (currently only Google is
     guarded; Mistral positional bug already fixed). HIGH RISK — see verification note.
  4. Relocate to_*/from_* wire serializers out of services/conversation.py into per-provider modules
     so the domain model stops knowing about every vendor (chaining-id collapse already done).
  5. files.py consolidation — ByteDocumentFile base for the 4 byte-backed doc classes; rename the
     `bytes` attribute (shadows the builtin) to `data`; rename from_url→from_path; add a timeout to
     from_web_url.
  6. Tools hardening — SSH host-key verification (RejectPolicy/known_hosts) and a command allow-list
     for the PowerShell/SSH tools. NOTE: changing the host-key policy could break my existing
     AutoAddPolicy setups — ASK me before changing default SSH behavior.

  HOW TO WORK (constraints):
  - API Keys are in the file `notebooks\.env`. Verify everything.
  - Verification gotcha: the repo root has a `types.py` that shadows stdlib `types` when cwd is the
    repo root. Run python from a NON-repo-root cwd (e.g. /tmp) with PYTHONPATH=/mnt/c/Programs so
    `import llm_platform` resolves. The venv is .venv (python 3.12); all provider SDKs are installed
    there EXCEPT paramiko.
  - For anything Anthropic-API-specific, the `claude-api` skill is available and authoritative.
  - After each structural change, update docs/technical_documentation.md (CLAUDE.md directive), and
    keep changes simple/pythonic.

  BEFORE YOU START EDITING: read the docs + memory, check git state, then give me a short plan that
  (a) confirms which deferred items are safely doable without live-provider testing and which you'd
  want me to test live, and (b) sequences them low-risk-first. For item 6, ask me about the SSH
  host-key policy. Work phase-by-phase with verification after each, and tell me clearly at the end
  what's verified vs. what still needs a live-provider check.