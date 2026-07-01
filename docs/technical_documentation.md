# LLM Platform Technical Documentation

Version: 2026-07-01
Source of truth: current implementation in this repository (`core/`, `adapters/`, `services/`, `helpers/`, `tools/`, `models_config.yaml`)

## 1. Purpose and scope
This project is a provider-agnostic Python platform for:
- Multi-provider LLM text interactions through one facade (`APIHandler`)
- Tool/function calling during conversations
- Multimodal input handling (text, images, audio, PDF, Excel, Word, PowerPoint, video)
- Provider-specific image, audio, document, and video input handling
- YAML-driven model routing and parameter governance

This document describes the implementation as it exists now.

## 2. High-level architecture
Main runtime flow:
1. Client code calls `core.llm_handler.APIHandler`.
2. `APIHandler` appends user input to `Conversation` (`services.conversation`).
3. `APIHandler` normalizes `additional_parameters` using `models_config.yaml` via `helpers.model_config.ModelConfig`.
4. `APIHandler` resolves adapter by model (`adapter` field in YAML) and lazily initializes adapter.
5. Adapter converts `Conversation` into provider-native payload and calls provider SDK/API.
6. Adapter converts provider output back into `Message`, including usage, reasoning/thinking, files, and function call metadata.
7. Message is appended to conversation state.

Primary modules:
- `core/llm_handler.py`: orchestration facade and entrypoint
- `core/parameter_normalizer.py`: `ParameterNormalizer` — normalizes `additional_parameters` against a model's YAML schema (extracted from the facade)
- `adapters/adapter_base.py`: `AdapterBase` contract plus provider-agnostic helpers shared by the adapters (parameter merge, usage extraction, callable→JSON-schema, tool-name resolution `_tool_name`, content-block formatting, PDF/tool-round constants). `load_dotenv()` runs once at module import (not per adapter construction)
- `adapters/openai_compatible_adapter.py`: `OpenAICompatibleAdapter` base for OpenAI-compatible providers (DeepSeek, OpenRouter, Z.AI)
- `adapters/*.py`: provider-specific translation and API calls
- `services/conversation.py`: provider-agnostic conversation/message/function-call domain model + platform-internal persistence (provider wire serialization lives in `adapters/serializers.py`)
- `services/files.py`: file abstractions and format conversion/extraction
- `helpers/model_config.py`: YAML model registry loader (cached + name-indexed) and parameter schema normalization
- `tools/*.py`: tool abstraction and built-in tool implementations (`SSHCommandTool` base for SSH admin tools)
- `types.py`: typed definition of supported additional parameters

## 3. Core facade (`APIHandler`)
File: `core/llm_handler.py`

### 3.1 Responsibilities
- Adapter registry and lazy adapter initialization
- Conversation state ownership (`self.the_conversation`)
- Parameter normalization (`_prepare_additional_parameters`)
- Sync and async request routing

### 3.2 Public API
- `request(model, prompt, functions=None, files=[], tool_output_callback=None, additional_parameters=None, **kwargs) -> Message`
- `request_async(...) -> Message`
- `request_llm(...) -> Message` (internal/public dispatch, no user-message append)
- `request_llm_async(...) -> Message`
- `calculate_tokens(text) -> {'bytes': int, 'tokens': int}` (tiktoken `cl100k_base`)

### 3.3 Adapter resolution
Adapter class is selected by model's `adapter` in `models_config.yaml` and imported + instantiated on demand. Adapter classes are registered as `"module:ClassName"` import paths (`ADAPTER_IMPORT_PATHS`) and loaded lazily through `importlib`, so importing `APIHandler` does not transitively import every provider SDK. Each adapter constructs its provider SDK client lazily: `AdapterBase` owns the single cached `client` property and delegates the one-line construction to a `_build_client()` hook each adapter overrides (OpenAI additionally has its own `async_client` property). So adapter construction needs no API key and performs no network/SDK work until the first request — only the selected provider's SDK is imported, and only when actually used.

Registered adapter classes include:
- `OpenAIAdapter`
- `AnthropicAdapter`
- `GoogleAdapter`
- `GrokAdapter`
- `DeepSeekAdapter`
- `OpenRouterAdapter`
- `MistralAdapter`
- `ZaiAdapter`

### 3.4 Additional parameter normalization pipeline
Implemented in `core/parameter_normalizer.ParameterNormalizer.normalize` (the facade owns a `ParameterNormalizer` and delegates via `_prepare_additional_parameters`):
1. Merge user `additional_parameters` and deprecated `**kwargs` (kwargs only fill missing keys).
2. Load model parameter definitions from YAML.
3. Apply defaults for definitions with `send_default: true` (including `max_tokens`, which is a standard YAML `additional_parameter` per model with provider-specific `request_key` mapping, e.g. `max_output_tokens` for OpenAI and Google).
4. Map friendly keys to nested request keys (`request_key`, e.g. `reasoning_effort -> reasoning.effort`, `max_tokens -> max_output_tokens`).
5. Drop fields where `include_in_request: false`.
6. Filter unsupported keys for the selected model and log warnings.

Normalization runs exactly once per call. `request` / `request_async` append the user message and forward the raw `additional_parameters` (and any deprecated `**kwargs`) to `request_llm` / `request_llm_async`, which are the single normalization point; direct callers of `request_llm` are therefore normalized identically.

## 4. Conversation domain model
File: `services/conversation.py`

Classes:
- `Conversation`: message list + system prompt + usage aggregations + serialization
- `Message`: role/content/files/usage/thinking/tool calls/tool responses
- `FunctionCall`: normalized function invocation metadata
- `FunctionResponse`: normalized tool output, optional files parsing
- `ThinkingResponse`: normalized reasoning/thinking content

Message roles: `user`, `assistant`, `function`

The domain model is provider-agnostic: it carries no vendor knowledge. Provider wire (de)serialization for `FunctionCall` / `FunctionResponse` / `ThinkingResponse` lives in `adapters/serializers.py` as standalone functions (e.g. `function_call_to_openai`, `function_call_to_anthropic`, `function_call_from_openai`, `function_call_from_grok`), keeping `services/` free of provider formats.

### Serialization
- Platform-internal persistence only: `Conversation.save_to_json()` serializes messages and file payloads, and `Conversation.read_from_json(data)` reconstructs conversation and supported file objects.
- Vendor wire (de)serialization is not part of the domain model; it lives in `adapters/serializers.py` (see §16).

## 5. File abstraction model
File: `services/files.py`

### 5.1 File typing
`define_file_type(file_name)` classifies: `text`, `pdf`, `excel`, `word`, `powerpoint`, `image`, `audio`, `video`, `unknown`.

### 5.2 Class hierarchy
- `BaseFile`
- `DocumentFile`
  - `TextDocumentFile` (text-backed: stores `self.text`)
  - `ByteDocumentFile` (byte-backed base: stores raw `self.data`, shares `from_bytes`/`from_file`)
    - `PDFDocumentFile`
    - `ExcelDocumentFile`
    - `WordDocumentFile`
    - `PowerPointDocumentFile`
- `MediaFile`
  - `ImageFile`
  - `AudioFile` (auto-converts non-mp3 input to mp3)
  - `VideoFile`

The byte-backed subclasses differ only in their `text` extraction property; the common storage (`self.data`) and constructors (`from_bytes(data, file_name="")`, `from_file(name)`) live in `ByteDocumentFile`. `MediaFile` and its subclasses store raw bytes in `self.data` as well; the local-filesystem reader is `MediaFile.from_path` and the network fetch `MediaFile.from_web_url` uses a 30s timeout. Across all byte-backed files the attribute is named `data` (not the shadowing builtin `bytes`); `base64`/`size`/`extension`/`text` remain the public read surface used by adapters and serialization.

### 5.3 Content extraction behavior
- PDF: text extraction via `PyPDF2`
- Excel: sheet text extraction via `pandas`
- Word (`.docx`): OOXML XML extraction from zip
- PowerPoint (`.pptx`): OOXML slide text extraction from zip

## 6. Model registry and metadata
File: `helpers/model_config.py`, config in `models_config.yaml`

### 6.1 Current catalog summary
- Total models: 22
- Visible models: 15
- Adapter families: 8

Models are grouped by `adapter`, with metadata:
- `name`, `display_name`
- `inputs`, `outputs`
- `pricing`
- `context_window`
- `visible`
- `additional_parameters` schema (including request defaults such as `max_tokens`)
- optional `background_mode`
- optional `agent_type` for Google managed-agent routing: `deep_research` (Gemini Deep Research) or `antigravity` (Antigravity agent — Interactions call into a remote sandbox). Both pair with `background_mode: true` to run asynchronously and be polled until terminal
- optional `adaptive_thinking` (Anthropic): when true the adapter sends `thinking: {type: "adaptive"}` + `output_config.effort`; otherwise it falls back to the legacy `thinking: {type: "enabled", budget_tokens}`. Set on models that reject `enabled`/`budget_tokens` (Opus 4.7/4.8, Sonnet 4.6)
- optional `uses_thinking_level` (Google): when true the adapter sends Gemini 3's categorical `thinking_level`; otherwise the legacy numeric `thinking_budget`. Set on the `gemini-3.x` chat models (replaces the former `"gemini-3" in model` substring check)
- optional `structured_output_with_tools` (Grok): when true, structured output may be combined with tools; otherwise that combination raises. Set on the Grok 4 family (replaces the former `model.startswith("grok-4")` check)
- optional `suppress_temperature` (OpenAI-compatible): when true the adapter drops the `temperature` parameter. No model currently sets it (the former `deepseek-reasoner` special case was removed), but it remains the per-model extension point should a temperature-rejecting model be added

These per-model capability flags follow the `adaptive_thinking` precedent: enabling a behavior is a `models_config.yaml` change rather than a hardcoded model-name check in adapter code. Model `inputs` doubles as a capability signal — e.g. OpenAI-compatible audio input is now gated on `"audio" in inputs` rather than a model-name check.

### 6.2 Parameter schema capabilities
`Model` normalizes `additional_parameters` and supports:
- type normalization (`string`, `enum`, `boolean`, etc.)
- default UI metadata (`ui`, `label`)
- option normalization (including ratio-like values)
- request mapping (`request_key`)
- flags: `send_default`, `include_in_request`

## 7. Adapter capability matrix

### 7.1 Text and tool orchestration adapters
All tool-calling loops (OpenAI sync/async, Anthropic, Google, Grok, Mistral, Z.AI) are bounded by the shared `MAX_TOOL_ROUNDS` constant (40, in `adapter_base.py`); exceeding it raises `RuntimeError` instead of recursing unboundedly.

- `OpenAIAdapter`
  - Responses API based chat flow
  - Sync + async request methods
  - Tool calling with recursive loop
  - Supports `web_search`, `code_execution`, structured output parsing, reasoning/text parameter pass-through
  - Background-mode models (`background_mode: true`) are polled to completion on both the sync and async paths (`_poll_background_response` / `_poll_background_response_async`)
  - Image-generation output is parsed as a single base64 string per `image_generation_call` (the Responses API `result` field)
  - Supports file citations retrieval from container files. `_parse_response` is pure (no network IO): it returns container-file citations as metadata, which `request_llm`/`request_llm_with_functions` then fetch via `_retrieve_container_files` (sync) and the async paths via `_retrieve_container_files_async` (async client), so parsing is testable and the async path never blocks on a synchronous fetch
- `AnthropicAdapter`
  - Sync only
  - Non-streaming and streaming execution paths
  - Streaming auto-enabled for large `max_tokens` (>= 21000)
  - Recursive tool-use loop
  - Supports web search, code execution, reasoning controls, structured output (non-streaming; when the configured `max_tokens` would force streaming, it is capped to 20 000 with a warning so structured output still works with the YAML defaults)
  - Tool lookup supports both `BaseTool` instances and plain callables (via `AdapterBase._tool_name`); a tool call whose name is not found is answered with an error `tool_result` instead of being dropped (dropping it made the model retry forever)
  - Thinking mode is chosen per model from the `adaptive_thinking` flag in `models_config.yaml`: flagged models (Opus 4.7/4.8, Sonnet 4.6) use `thinking: {type: "adaptive"}` + `output_config.effort`; others use legacy `thinking: {type: "enabled", budget_tokens}`. (Models such as Opus 4.8 reject `enabled`/`budget_tokens` with a 400.)
  - Automatic prompt caching is always on: `_prepare_request_kwargs` sets a single top-level `cache_control: {type: "ephemeral"}` (applied to every request path), so the stable system + tools + history prefix is served from cache across turns and tool-use loops. Prompts below the model's minimum cacheable length are silently left uncached. Usage reports the cache breakdown in `cache_read_tokens` / `cache_creation_tokens`, and `prompt_tokens` is the full input (uncached + cache read + cache write) since the API's `input_tokens` counts only the uncached remainder when caching is active.
  - Performs max-token correction against context window
- `GoogleAdapter`
  - Built entirely on the Gemini **Interactions API** (`client.interactions.create`); the legacy `client.models.generate_content` surface is no longer used
  - Conversation history is converted into the Interactions `step_list` input array: every entry is a typed Step — `user_input` / `model_output` (each carrying a `content` array) for plain exchanges, plus `function_call` / `function_result` for prior tool round-trips. Legacy role-keyed Turn objects (`{"role": ..., "content": [...]}`) are rejected by the steps-based API.
  - System prompt is sent as the top-level `system_instruction` parameter; tools, system instructions, and `generation_config` are re-supplied on every call (interaction-scoped per the API contract)
  - Server-side conversation state is reused across turns via `previous_interaction_id`. The first turn sends the full converted history; the returned `interaction.id` is stored on the assistant `Message`. On every subsequent turn the adapter resolves the prior id through `Conversation.previous_interaction_id_for_google` and sends only the new `user_input` (mirroring the `previous_response_id_for_openai` pattern used by `OpenAIAdapter`).
  - Function-calling round-trips inside a single user turn use the same `previous_interaction_id` chaining; only the new `function_result` entries are sent on follow-up calls
  - Tools are emitted as plain dicts: `{"type": "function", ...}` for `BaseTool` declarations plus `{"type": "google_search"}`, `{"type": "url_context"}`, `{"type": "code_execution"}` for built-ins
  - Generation parameters (`temperature`, `max_output_tokens`, `thinking_level` for models flagged `uses_thinking_level` / `thinking_budget` otherwise, `thinking_summaries: "auto"`) go inside `generation_config`. Structured output is sent through the top-level `response_format` field (the Interactions API polymorphic shape) via `extra_body` to bypass stale SDK serialization: `{type: "text", mime_type: "application/json", schema}`.
  - Responses are parsed off `interaction.steps`: `model_output` → text/images/citations, `thought` → `ThinkingResponse`, `function_call` → `FunctionCall`, `code_execution_call` / `code_execution_result` → `additional_responses`
  - Routes Gemini models marked with both `background_mode: true` and `agent_type: deep_research` to a separate Deep Research path (`agent=<model>`, `background=True`, `store=True`), polled until terminal status and parsed into a standard cited `Message`
  - Supports Deep Research text/image/PDF/audio/video inputs from the latest user message
  - Routes models marked `agent_type: antigravity` to the Antigravity managed-agent path (`_request_antigravity`): `interactions.create(agent=<model>, environment="remote", ...)` provisions a remote Linux sandbox and runs the agent's tool-use loop (code execution, web search, URL fetch, filesystem) server-side. The agent rejects `generation_config`/structured output, so neither is sent — only `system_instruction`, built-in tools (per the `web_search`/`code_execution`/`url_context` flags), and custom functions. Built-in and filesystem calls are executed by the sandbox; only *custom* functions need a client-side round-trip, fed back via `previous_interaction_id` (stateful-only function calling) reusing the same `environment`. When the model is flagged `background_mode: true` (the default for `antigravity-preview-05-2026`) every `interactions.create` runs with `background=True` + `store=True` and is polled by `_poll_agent_interaction` until a terminal status or `requires_action` (the agent waiting on a custom-function result) — the recommended mode for these long-running agent tasks. Responses are parsed with the shared `_parse_interaction_response` (which falls back to `interaction.output_text` when no `model_output` step text is present)
- `GrokAdapter`
  - Sync chat with optional tool execution loop
  - Supports web search and code execution tools in xAI SDK
  - Supports structured output through xAI SDK `response_format` for both standard requests and tool-enabled requests on models flagged `structured_output_with_tools` (the Grok 4 family)
- `MistralAdapter`
  - Sync chat and recursive function-calling (`tool_choice: "auto"`, bounded by `MAX_TOOL_ROUNDS`); the final assistant message is appended and returned by the tool loop itself
  - Tool-call history is serialized in Chat Completions shape (`function_call_to_openai_chat` / `function_response_to_openai_chat`, parsed back via `function_call_from_openai_chat`), matching the OpenAI-compatible adapters
  - Request parameters (`temperature`, `max_tokens`, passthrough keys, filtered by `MISTRAL_RESERVED_KEYS`) are applied on both the plain-chat and function-calling paths
- `DeepSeekAdapter`, `OpenRouterAdapter`, `ZaiAdapter`
  - Thin subclasses of `OpenAICompatibleAdapter`. `DeepSeekAdapter` / `OpenRouterAdapter` declare only `BASE_URL` / `ENV_VAR` and use the OpenAI client against the provider base URL; `ZaiAdapter` (GLM models) additionally overrides `_build_client` to use the official `zai-sdk` `ZaiClient`, which exposes the same OpenAI-compatible `chat.completions.create` surface. Temperature suppression is driven by the per-model `suppress_temperature` flag in the base rather than a name-based override
  - Shared OpenAI-compatible chat path: text/image/audio/document conversion, parameter marshalling, and usage extraction live in the base. Tool-call history is serialized in Chat Completions shape (`function_call_to_openai_chat` / `function_response_to_openai_chat`): tool calls nested under `tool_calls[].function` on the assistant message, tool results sent as standalone `role: "tool"` messages
  - `DeepSeekAdapter` / `OpenRouterAdapter` do not implement tool calling: they inherit the uniform `AdapterBase.request_llm_with_functions` that raises `NotImplementedError`
  - `ZaiAdapter` adds tool calling and web search on top of the shared base:
    - **Function calling**: `request_llm` routes to a recursive `request_llm_with_functions` loop (request → execute local `BaseTool`/callable tools → append `FunctionCall`/`FunctionResponse` records → re-ask) until the model stops emitting `tool_calls`. Function tools are emitted as `{"type": "function", "function": {...}}` via `_convert_function_to_tool` (reusing `BaseTool.to_params(provider="openai")` or `_callable_to_json_schema`)
    - **Web search**: Z.AI's built-in server-side `web_search` tool, enabled by the `web_search` additional parameter. `_build_request_params` is overridden to attach the built-in tool (`{"type": "web_search", "web_search": {"enable": True, "search_engine": "search-prime", "search_result": True}}`) so both the plain-chat and function-calling paths pick it up. No static `search_query` is sent — GLM derives queries from the conversation. Built-in and function tools are merged on the same request

### 7.2 Async support
`AdapterBase` provides a default `request_llm_async` that runs the adapter's synchronous `request_llm` off the event loop via `asyncio.to_thread`. As a result `APIHandler.request_async` / `request_llm_async` work for every adapter rather than only OpenAI.

`OpenAIAdapter` overrides the default with a native async implementation (`request_llm_async`, `request_llm_with_functions_async`) backed by the async OpenAI client; all other adapters inherit the thread-offloaded default.

## 8. Multimodal behavior by adapter (implemented)
- OpenAI: text, image, audio, document inputs
- Anthropic: text/image/document
- Google: text/image/audio/document/video inputs; Gemini Deep Research agents through background Interactions API calls; Antigravity managed agent (`agent_type: antigravity`) over the Interactions API with a remote sandbox, accepting text/image input only
- Grok: text/image/document in chat
- Mistral: text/image/document chat
- DeepSeek/OpenRouter: text + image/document conversion (OpenAI-compatible payload)
- Z.AI (GLM-5.2): text (OpenAI-compatible payload via `zai-sdk` `ZaiClient`); supports function calling and the built-in `web_search` tool

## 9. Tools subsystem
Files: `tools/base.py` and concrete tools in `tools/*.py`

### 9.1 Base abstraction
`BaseTool` requires:
- callable interface (`__call__(...)`)
- nested `InputModel` Pydantic schema

`BaseTool.to_params(provider=...)` emits provider-specific tool declarations for:
- OpenAI
- Anthropic
- Google (schema transformed to Gemini-compatible form; `GoogleAdapter` wraps the resulting dict with `{"type": "function", ...}` to satisfy the Interactions API tools schema)
- Grok (the `xai_sdk` `tool` type is imported lazily inside `to_params`, so importing the tools layer does not require `xai_sdk`)

Plain Python callables passed as tools are converted to a JSON schema once by `AdapterBase._callable_to_json_schema`; each adapter wraps that canonical schema in its provider-specific envelope.

### 9.2 Built-in tool modules
- `RunPowerShellCommand` (persistent PowerShell process)
- `CzechLaws`
- `Reddit`
- `RaspberryAdmin`, `UbuntuAdmin` (thin subclasses of `SSHCommandTool` in `tools/ssh_command.py`)

### 9.3 Command-execution hardening
- `SSHCommandTool` imports `paramiko` lazily (inside `__call__`), so importing the tools layer does not require `paramiko` to be installed.
- Command-executing tools (`SSHCommandTool`, `RunPowerShellCommand`) accept an optional `allowed_commands` constructor argument. When provided, `BaseTool._check_command_allowed` first rejects any command containing a shell control operator (`; | & \` $ > < ( )` or a newline) — so chaining/piping/substitution cannot smuggle a non-allowed program past the check — and then requires the command's leading token to be in the allow-list; either failure raises `PermissionError` before execution. The default `None` applies no restriction, preserving existing behavior.
- SSH host-key policy remains `AutoAddPolicy` (unchanged by design).

## 10. Environment variables and credentials
Current code expects:
- `OPENAI_API_KEY` (OpenAI SDK default)
- `ANTHROPIC_API_KEY`
- `GOOGLE_GEMINI_API_KEY`
- `XAI_API_KEY`
- `DEEPSEEK_API_KEY`
- `OPENROUTER_API_KEY`
- `MISTRAL_API_KEY`
- `ZAI_API_KEY`

Important: current `README.md` and some older docs list different variable names for Google/Grok. The values above reflect actual adapter code.

## 11. Dependencies
From `requirements.txt`:
- Provider SDKs: `openai`, `anthropic`, `google-genai`, `mistralai`, `xai_sdk`, `zai-sdk`
- Data/media: `pandas`, `pillow`, `PyPDF2`, `pydub`, `lxml`
- Tooling and support: `pydantic`, `python-dotenv`, `PyYAML`, `requests`, `tiktoken`, `loguru`, `rich`, `praw`

## 12. Error handling and observability
- Logging uses `loguru` in orchestration and adapters.
- `APIHandler.request_llm` / `request_llm_async` let adapter exceptions propagate to the caller, consistently across the sync and async paths; they no longer swallow exceptions into a fabricated assistant message appended to the conversation.
- `APIHandler.get_adapter` raises `ValueError` for a model not present in `models_config.yaml`.
- Some adapter methods remain `NotImplemented` and will raise directly.

## 13. Known implementation gaps and inconsistencies
1. Tool-calling support is partial across adapters (fully implemented in OpenAI/Anthropic/Google/Grok/Mistral/Z.AI, not in DeepSeek/OpenRouter).
2. Mutable default arguments still exist in the `Message` initializer (`[]` defaults); the adapter and `APIHandler` method signatures that previously shared this pattern have been migrated to `None` defaults.
3. The `README.md` environment-variable list now matches the adapter code (`GOOGLE_GEMINI_API_KEY`, `XAI_API_KEY`).

## 14. Request lifecycle details

### 14.1 Standard chat call
1. Client calls `APIHandler.request(...)`.
2. User `Message` appended to `Conversation`.
3. Additional parameters normalized against model schema.
4. Adapter selected by model.
5. Adapter converts history and calls provider.
6. Provider output parsed into assistant `Message`.
7. Assistant message appended and returned.

### 14.2 Tool-calling flow
1. Adapter sends tool definitions + conversation.
2. Provider returns tool call(s).
3. Adapter resolves and executes local tool callable(s).
4. Tool outputs are captured as `FunctionResponse` records.
5. Conversation is updated with tool call/response records.
6. Adapter recursively calls provider until final non-tool assistant output is produced, bounded by `MAX_TOOL_ROUNDS` (40); exceeding the bound raises `RuntimeError`.

### 14.3 Gemini Deep Research flow
1. Client calls a model such as `deep-research-preview-04-2026` or `deep-research-max-preview-04-2026`.
2. `models_config.yaml` marks the model with `background_mode: true` and `agent_type: deep_research`, so `GoogleAdapter` uses the Gemini Interactions API instead of `models.generate_content`.
3. The latest user message is converted into Interactions input; the conversation system prompt is prepended to the text input because Deep Research agents do not support `system_instruction`.
4. Images, PDFs, audio, and video are sent as inline base64 content while Office/text documents are converted to text content.
5. The adapter starts the interaction with `agent=<model>`, `background=True`, and `store=True`; it does not send `generation_config` because Gemini agents require agent-specific configuration through `agent_config`.
6. The adapter polls until the interaction reaches a terminal status (`completed`, `failed`, `cancelled`, or `incomplete`).
7. The completed interaction is parsed using the May 2026 steps schema: the adapter walks `interaction.steps`, picks `model_output` steps, and pulls text / image / annotation items out of each step's `content[]` array. Text content joins into the assistant message body, image content becomes `ImageFile` attachments, citation annotations become `additional_responses`, and `interaction.usage` (`total_input_tokens`, `total_output_tokens`, `total_tokens`) is mapped to the usual usage keys.

## 15. Extending the platform

### 15.1 Add a new model
1. Add entry to `models_config.yaml` with:
   - `name`, `adapter`, `inputs`, `outputs`, token/context limits
   - optional `additional_parameters` definitions
2. Ensure the mapped adapter exists in `APIHandler._lazy_initialization_of_adapter`.

### 15.2 Add a new adapter
1. Implement adapter class in `adapters/` inheriting `AdapterBase`.
2. Implement at least `request_llm` and conversation conversion.
3. Add adapter mapping in `APIHandler` lazy-init map.
4. Add model entries in `models_config.yaml`.

### 15.3 Add a new tool
1. Subclass `BaseTool`.
2. Provide Pydantic `InputModel`.
3. Implement `__call__`.
4. Pass tool instance in `APIHandler.request(..., functions=[...])`.

## 16. File and package map
- `core/llm_handler.py`: orchestration facade (lazy adapter registry, conversation state, sync/async routing)
- `core/parameter_normalizer.py`: `ParameterNormalizer` parameter pipeline
- `helpers/model_config.py`: YAML model registry (cached + name-indexed) and parameter normalization
- `services/conversation.py`: provider-agnostic conversation and tool metadata classes (no vendor knowledge); platform-internal persistence only (`save_to_json`/`read_from_json`)
- `adapters/serializers.py`: provider wire (de)serialization for the domain objects (functions like `function_call_to_openai` for the OpenAI Responses API and `function_call_to_openai_chat` for the OpenAI-compatible Chat Completions API), kept out of the domain model so `services/` stays provider-agnostic
- `services/files.py`: file classes and text/media extraction
- `adapters/adapter_base.py`: `AdapterBase` contract + shared adapter helpers
- `adapters/openai_compatible_adapter.py`: `OpenAICompatibleAdapter` base (DeepSeek, OpenRouter, Z.AI)
- `adapters/zai_adapter.py`: `ZaiAdapter` — Z.AI GLM models via the official `zai-sdk` `ZaiClient` (OpenAI-compatible); adds function calling (recursive tool loop) and the built-in `web_search` tool
- `adapters/*.py`: provider integrations
- `tools/base.py`: `BaseTool` contract + per-provider declaration emission
- `tools/ssh_command.py`: `SSHCommandTool` base for SSH admin tools
- `tools/*.py`: callable tool implementations
- `models_config.yaml`: model routing and metadata
- `types.py`: typed `AdditionalParameters`
