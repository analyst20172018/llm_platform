# LLM Platform Technical Documentation

Version: 2026-09-08
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
- `adapters/response_metadata.py`: provider status/finish normalization and citation/hosted-result extraction; provider-specific interpretation remains outside the domain model
- `adapters/openai_compatible_adapter.py`: `OpenAICompatibleAdapter` base for OpenAI-compatible providers (DeepSeek, OpenRouter, OrcaRouter, Z.AI, Kimi); includes opt-in sync/async local-tool loops and JSON output marshalling
- `adapters/json_output.py`: shared JSON-mode/schema envelopes and system instructions for Mistral, DeepSeek, and OpenRouter
- `adapters/wiro_ai_adapter.py`: `WiroAIAdapter` for WiroAI's asynchronous Run/Task HTTP API
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
- `OrcarouterAdapter`
- `WiroAIAdapter`
- `MistralAdapter`
- `ZaiAdapter`
- `KimiAdapter`

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

### Serialization and continuation
- `Conversation.save_to_json()` returns a detached, JSON-compatible dictionary with `version: 2`. `read_from_json()` accepts this format and legacy unversioned files; unknown versions raise `ValueError`.
- Persistence preserves message IDs, timestamps, provider/model identity, native `provider_data`, thinking, usage, `additional_responses` (including citation metadata), every tool call/result's distinct `id` and `call_id`, and namespaced tool metadata such as OpenAI's `caller`.
- Text documents retain text. All supported binary documents/media retain original bytes and concrete file classes, including Word, PowerPoint, and Video. Saving does not extract Office/PDF text. Restoring audio does not transcode bytes a second time, even if its original name has a non-MP3 extension. Tool-result construction and persistence do not mutate the source result dictionaries.
- `Conversation.checkpoint(provider, model, response_id, **state)` records the length and SHA-256 fingerprint of the local prefix represented by remote state, including the system prompt and attachments. `continuation(provider, model)` returns state only when that prefix still matches. OpenAI and Google send every message after the checkpoint, including unsent user messages, intervening exchanges from other providers, and tool results. Model/provider changes cannot reuse foreign IDs. `last_assistant_id` remains a legacy display helper only.
- Editing, deleting, inserting, reordering, or changing files within a saved prefix invalidates its checkpoint. `clear()` clears messages and checkpoints. `reset_continuation(provider, model)` explicitly discards the chosen remote reference; for Antigravity this also starts a new environment on the next call.
- Adapters append assistant tool-call messages and checkpoint them before appending newly executed results. Google records results in separate `function` messages, so an unsent result never becomes part of the remote checkpoint. All provider converters support these standalone tool-result records.
- OpenAI and Google retry an explicitly missing/deleted/expired continuation once with full compatible history. The retry is limited to state-related 400/404/410 responses, identified by the continuation field or referenced ID; model, authentication, rate-limit, and unrelated failures are not retried by this layer. Antigravity retries keep the selected environment, whose availability still depends on the provider. Failed retries preserve local state. OpenAI `store=False` requests replay full history and do not create a new remote checkpoint.

### Native response replay
`Message.provider_data` holds detached provider JSON alongside the normalized display content. `Message.replay_data(provider, model)` returns a copy only for a compatible origin and unchanged normalized content. Editing an assistant message falls back to its edited normalized representation rather than replaying stale native content. Tool results are independent of the assistant's native output and are serialized after it.

Provider wire conversion stays in the adapters; the domain model only stores provider names, opaque JSON, and fingerprints:
- OpenAI retains every output item in order, including encrypted reasoning, hosted tools, citations, exact function arguments, and multi-agent attribution. The adapter never synthesizes reasoning items from display summaries. SDK-only `parsed` and `parsed_arguments` conveniences are retained in persistence but removed from wire replay. Function outputs preserve the original `call_id` and optional `caller`.
- Google retains complete Interactions steps, including thought signatures and hosted-tool signatures. As required by Interactions, native steps can be replayed across Gemini models; continuation IDs remain scoped by model. The adapter never forwards these steps to another provider.
- Anthropic retains the complete ordered content blocks on streaming and non-streaming paths, including redacted thinking, thinking signatures, citations, and hosted tools. Container identity is retained and reused for the same model. Display thinking is never turned into a fabricated signed block. A bounded request loop resumes `pause_turn` with those exact blocks and the same container; every paused round remains in conversation history.
- OpenRouter and the shared Chat Completions adapters retain exact assistant content/tool calls plus native reasoning fields. OpenRouter's visible `reasoning` is exposed as thinking while opaque `reasoning_details` remain intact. Kimi and Z.AI replay reasoning only for its recorded provider/model.
- Mistral separates chunked answer text and thinking for display and keeps the original chunks for subsequent requests.
- Grok uses the installed SDK's response-to-history conversion to retain encrypted reasoning, tool calls/results, and citation metadata.

Regression coverage is in `tests/test_continuation_persistence.py`, with mocked sync/async requests, real JSON round trips, SDK-backed Mistral/Grok replay checks, history mutation, provider switching, missing-state fallback, and Antigravity environment reuse/reset. These are offline checks, not live provider acceptance tests.

Contracts checked through the MCP servers in `.mcp.json`: [OpenAI continuation and replay](https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling#continue-after-client-owned-function-calls), [Gemini thought signatures](https://ai.google.dev/gemini-api/docs/thinking#thought-signatures), and [Antigravity environments](https://ai.google.dev/gemini-api/docs/antigravity-agent#environments).

### Result status, citations, and hosted results

`Message` adds these optional fields, preserved by version-2 JSON persistence:
- `status`: normalized `completed`, `incomplete`, `refused`, `failed`, `cancelled`, `requires_action`, `paused`, `queued`, `in_progress`, or `unknown`. Missing/legacy or unfamiliar finish metadata defaults to `unknown`; it never implies success.
- `finish_reason`: the original provider finish/stop reason, or Responses/Interactions status. `error` and `incomplete_details` retain detached native details when supplied.
- `citations`: dictionaries with `provider`, `url`, `title`, `start_index`, `end_index`, `source`, and `location`. `source` keeps the entire native annotation/reference, including file IDs, document/page ranges, encrypted source references, and any additional provider fields. `location` identifies the original block/content index where available. Span offsets retain the provider's units and refer to the original block, not concatenated `Message.content`; unavailable fields are `None`.
- `hosted_tool_results`: detached native hosted call/result records, including IDs, code/search output, errors, and generated-file references. They are not local `FunctionResponse` records. OpenAI code-interpreter source is exposed here instead of appended to answer text. Existing container-file downloads and generated image attachments remain available in `files`.

Adapters normalize status before deciding whether to execute local calls. Explicit failed/truncated/refused/cancelled/pending/unknown states cannot execute tools. Legacy provider responses without a finish field can still request action through explicit function calls. `Message.can_execute_tools` is true only for `requires_action`. Terminal provider failures return a message with partial text/files and native metadata, including Gemini Deep Research (which previously raised for non-completed status). HTTP/SDK exceptions and Wiro Run task failures continue to propagate.

Metadata describes each individual response. Intermediate tool/paused rounds retain their own citations, results, and usage in `Conversation.messages`; the returned message describes the final round. New display metadata is excluded from native replay fingerprints. Existing saved messages lacking these fields load with safe defaults; older checkpoint fingerprints may invalidate safely and cause full-history replay.

OpenAI and Google polling and tool continuations, and Claude local-tool/pause continuation, use a monotonic deadline controlled by `adapter.RESPONSE_TIMEOUT_SECONDS` (default 1800 seconds). A deadline is shared across rounds, and polling sleeps are clamped to remaining time. `ResponseTimeoutError` (a `TimeoutError` subclass in `adapters/adapter_base.py`) exposes `response_id` and the last native `response` (a `ClaudeStreamProcessor` on Claude paths) where available; completed rounds already recorded in the conversation remain available. These are orchestration deadlines checked between SDK operations/stream events, not preemptive cancellation of a blocking SDK call or user callable; SDK transport timeouts still apply. No remote cancellation request is sent on timeout. The 40-round guard remains independent.

Offline regressions in `tests/test_tool_lifecycle.py` cover forced-tool termination, Claude hosted/local combinations and pause replay on sync/async streaming/non-streaming paths, round/deadline exhaustion, terminal tool guards, citations/results across JSON restore, and bounded background polling. Contracts were checked through the `.mcp.json` documentation servers for [OpenAI background polling](https://developers.openai.com/api/docs/guides/background) and [Gemini Interactions](https://ai.google.dev/gemini-api/docs/interactions), plus official [Claude pause/container handling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool#pause_turn-stop-reason) and [Kimi tool calling](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart). No live provider generation was used for validation.

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
- Total models: 23
- Visible models: 16
- Adapter families: 10

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
All tool-calling loops (OpenAI sync/async, Anthropic sync/async, Google sync/async, Kimi sync/async, Grok, Mistral, Z.AI) are bounded by the shared `MAX_TOOL_ROUNDS` constant (40, in `adapter_base.py`); exceeding it raises `RuntimeError` instead of recursing unboundedly.

- `OpenAIAdapter`
  - Responses API based chat flow
  - Sync + async request methods
  - Tool calling with recursive loop
  - The `web_search` toggle enables the current Responses API hosted tool (`{"type": "web_search"}`) on all request paths.
  - Supports `web_search`, `code_execution`, structured output parsing, reasoning/text parameter pass-through. GPT-5.6 pro mode is exposed as the `reasoning_mode` additional parameter (enum `standard`/`pro`, `request_key: reasoning.mode`, sent only when explicitly selected) — pure YAML/normalizer configuration, no adapter code. Structured output sets `text_format`, which only `responses.parse()` accepts, so all four request paths (sync/async, with/without tools) branch on `"text_format" in parameters` to choose `parse` vs `create` (covered by `tests/test_openai_structured_output.py`)
  - Multi-agent (beta, GPT-5.6 family) is exposed as the `agent_count` additional parameter (enum `0/2/3/5/8`, default `0` = single agent, `send_default: false`; the same friendly key `types.py` already defines for Grok's multi-agent models). When > 0, `_create_parameters_for_calling_llm` supplies `multi_agent: {enabled: true, max_concurrent_subagents: N}` and `betas: [responses_multi_agent=v1]`; `_create_response` / `_create_response_async` route these requests through the typed `client.beta.responses` surface (requiring `openai>=2.45.0`) so beta-only `agent` and `phase` attribution survives deserialization. Orchestration (spawning, messaging, waiting) is hosted by the Responses API, so function calls emitted by any agent in the tree are executed by the existing tool loop and continued via the standard `previous_response_id` path. Two guards remain: `reasoning.summary` is not requested when multi-agent is on (unsupported combination), and combining with `structured_output` raises `ValueError`. `_parse_response` skips subagent-attributed items and root messages outside the `final_answer` phase, ignores hosted orchestration items (`multi_agent_call`, `multi_agent_call_output`, `agent_message`), and suppresses repeated agent-message text so duplicate root final-answer items are returned once while distinct fragments are preserved (covered by `tests/test_openai_multi_agent.py`).
  - Background-mode models (`background_mode: true`) are polled to completion on both the sync and async paths (`_poll_background_response` / `_poll_background_response_async`)
  - Image-generation output is parsed as a single base64 string per `image_generation_call` (the Responses API `result` field)
  - Supports file citations retrieval from container files. `_parse_response` is pure (no network IO): it returns container-file citations as metadata, which `request_llm`/`request_llm_with_functions` then fetch via `_retrieve_container_files` (sync) and the async paths via `_retrieve_container_files_async` (async client), so parsing is testable and the async path never blocks on a synchronous fetch
- `AnthropicAdapter`
  - Sync + native async request methods (`request_llm_async` backed by a lazily constructed `anthropic.AsyncAnthropic` client; the async side mirrors the sync dispatch in two helpers — `_request_llm_simple_async` / `_request_llm_with_tools_async` — each taking a `stream` flag instead of separate streaming methods)
  - Non-streaming and streaming execution paths
  - Streaming auto-enabled for large `max_tokens` (>= 21000)
  - Local `tool_use` executes through `_handle_tool_calls` / `_handle_tool_calls_async`; the async helper awaits coroutine tools. Hosted `pause_turn` appends the complete assistant response and resumes without executing hosted calls locally. Both consume the same 40-round and elapsed-time budget. Search and code execution remain enabled alongside local functions on every round.
  - Token counting/max-token correction have async counterparts (`count_tokens_async` / `correct_max_tokens_async` on the async client) sharing the clamp logic (`_clamp_max_tokens`), so the async path never blocks the event loop
  - Supports web search, code execution, reasoning controls, structured output (on both the streaming and non-streaming paths)
  - Tool lookup supports both `BaseTool` instances and plain callables (via `AdapterBase._tool_name`); a tool call whose name is not found is answered with an error `tool_result` instead of being dropped (dropping it made the model retry forever)
  - Thinking mode is chosen per model from the `adaptive_thinking` flag in `models_config.yaml`: flagged models (Opus 4.7/4.8, Sonnet 4.6) use `thinking: {type: "adaptive"}` + `output_config.effort`; others use legacy `thinking: {type: "enabled", budget_tokens}`. (Models such as Opus 4.8 reject `enabled`/`budget_tokens` with a 400.)
  - Automatic prompt caching is always on: `_prepare_request_kwargs` sets a single top-level `cache_control: {type: "ephemeral"}` (applied to every request path), so the stable system + tools + history prefix is served from cache across turns and tool-use loops. Prompts below the model's minimum cacheable length are silently left uncached. Usage reports the cache breakdown in `cache_read_tokens` / `cache_creation_tokens`, and `prompt_tokens` is the full input (uncached + cache read + cache write) since the API's `input_tokens` counts only the uncached remainder when caching is active.
  - Performs max-token correction against context window
- `GoogleAdapter`
  - Built entirely on the Gemini **Interactions API** (`client.interactions.create`); the legacy `client.models.generate_content` surface is no longer used
  - Sync + native async request methods: `request_llm_async` runs on the SDK's native async surface (`client.aio`, exposed via the adapter's `async_client` property) and mirrors the sync dispatch — standard chat/tool loop (`_execute_function_calls_async`, which additionally awaits coroutine tools), Deep Research (`_request_deep_research_async` + `_poll_deep_research_interaction_async`), and Antigravity (`_request_antigravity_async` + `_poll_agent_interaction_async` with `asyncio.sleep` polling). Request building and response parsing are shared with the sync path (pure helpers, no I/O)
  - Conversation history is converted into the Interactions `step_list` input array: every entry is a typed Step — `user_input` / `model_output` (each carrying a `content` array) for plain exchanges, plus `function_call` / `function_result` for prior tool round-trips. Legacy role-keyed Turn objects (`{"role": ..., "content": [...]}`) are rejected by the steps-based API.
  - System prompt is sent as the top-level `system_instruction` parameter; tools, system instructions, and `generation_config` are re-supplied on every call (interaction-scoped per the API contract)
  - Server-side state is reused through provider/model checkpoints in `Conversation.continuations`. A valid checkpoint supplies `previous_interaction_id` and every local message after its recorded prefix. Missing or changed checkpoints send the full compatible history. OpenAI uses the same checkpoint mechanism with `previous_response_id`. Explicit missing/expired-state errors trigger one full-history replay attempt; unrelated errors propagate.
  - Function-calling round-trips inside a single user turn use the same `previous_interaction_id` chaining; only the new `function_result` entries are sent on follow-up calls
  - Tools are emitted as plain dicts: `{"type": "function", ...}` for `BaseTool` and plain-callable declarations plus `{"type": "google_search"}`, `{"type": "url_context"}`, `{"type": "code_execution"}` for built-ins
  - Generation parameters (`temperature`, `max_output_tokens`, `thinking_level` for models flagged `uses_thinking_level` / `thinking_budget` otherwise, `thinking_summaries: "auto"`) go inside `generation_config`. Structured output uses the named top-level `response_format` SDK argument: `{type: "text", mime_type: "application/json", schema}`. Pydantic models, raw schema dictionaries, and annotated types (including typed containers) retain their references, nullability, and constraints. The old `extra_body` workaround is removed; sync/async SDK serialization is covered offline.
  - Responses are parsed off `interaction.steps`: `model_output` → text/images/citations, `thought` → `ThinkingResponse`, `function_call` → `FunctionCall`, `code_execution_call` / `code_execution_result` → `additional_responses`
  - Routes Gemini models marked with both `background_mode: true` and `agent_type: deep_research` to a separate Deep Research path (`agent=<model>`, `background=True`, `store=True`), polled until terminal status and parsed into a standard cited `Message`
  - Deep Research uses the same checkpoint and delta selection as standard chat, preserving all unsent text/image/PDF/audio/video inputs. System instructions remain inline in the input; agent configuration is exposed through `agent_config`.
  - Routes models marked `agent_type: antigravity` to the Antigravity managed-agent path (`_request_antigravity`): `interactions.create(agent=<model>, environment="remote", ...)` provisions a remote Linux sandbox and runs the agent's tool-use loop (code execution, web search, URL fetch, filesystem) server-side. The agent rejects `generation_config`/structured output, so neither is sent — `system_instruction`, `agent_config` (including `max_total_tokens`), built-in tools (per the `web_search`/`code_execution`/`url_context` flags), and custom functions. Built-in and filesystem calls are executed by the sandbox; only *custom* functions need a client-side round-trip, fed back via `previous_interaction_id` (stateful-only function calling) reusing the same `environment`. Both the interaction ID and environment ID survive JSON persistence and are reused on subsequent user turns. Set `additional_parameters={"new_environment": True}` to reset the checkpoint and provision a fresh sandbox; `Conversation.reset_continuation("google", model)` is the equivalent explicit state operation. Replay retries retain the selected environment; an unavailable environment raises the provider error rather than silently creating a replacement. When the model is flagged `background_mode: true` (the default for `antigravity-preview-05-2026`) every `interactions.create` runs with `background=True` + `store=True` and is polled by `_poll_agent_interaction` until a terminal status or `requires_action` (the agent waiting on a custom-function result) — the recommended mode for these long-running agent tasks. Responses are parsed with the shared `_parse_interaction_response` (which falls back to `interaction.output_text` when no `model_output` step text is present)
- `GrokAdapter`
  - Sync chat with optional tool execution loop. SDK `ToolCall.type` distinguishes client functions from hosted calls, so server search/code calls are retained without being executed locally. Native finish reasons, URL/inline citations, and hosted call/result records are exposed on the returned message.
  - Supports web search and code execution tools in xAI SDK
  - Supports structured output through xAI SDK `response_format` for both standard requests and tool-enabled requests on models flagged `structured_output_with_tools` (the Grok 4 family)
- `MistralAdapter`
  - Sync chat and recursive function-calling (`tool_choice: "auto"`, bounded by `MAX_TOOL_ROUNDS`); the final assistant message is appended and returned by the tool loop itself
  - Tool-call history is serialized in Chat Completions shape (`function_call_to_openai_chat` / `function_response_to_openai_chat`, parsed back via `function_call_from_openai_chat`), matching the OpenAI-compatible adapters
  - Request parameters (`temperature`, `max_tokens`, passthrough keys, filtered by `MISTRAL_RESERVED_KEYS`) are applied on both the plain-chat and function-calling paths
  - `structured_output=True` enables `response_format={type: "json_object"}`. Pydantic classes, raw JSON Schemas, and native response-format envelopes use `json_schema` on `chat.complete`, with schema guidance added to detached system-message history. The same mapping applies to tool requests and the thread-offloaded async path; the response remains JSON text in `Message.content`.
- `DeepSeekAdapter`, `OpenRouterAdapter`, `OrcarouterAdapter`, `ZaiAdapter`
  - Thin subclasses of `OpenAICompatibleAdapter`. `DeepSeekAdapter`, `OpenRouterAdapter`, and `OrcarouterAdapter` configure their base URL, credentials, and provider-specific parameters and use the OpenAI client against the provider base URL; `ZaiAdapter` (GLM models) additionally overrides `_build_client` to use the official `zai-sdk` `ZaiClient`, which exposes the same OpenAI-compatible `chat.completions.create` surface. Temperature suppression is driven by the per-model `suppress_temperature` flag in the base rather than a name-based override
  - Shared OpenAI-compatible chat path: text/image/audio/document conversion, parameter marshalling, response→`Message` construction (`_message_from_response`), and usage extraction live in the base. A provider that returns `reasoning_content` on the assistant message has it captured as a `ThinkingResponse` (used by DeepSeek). Tool-call history is serialized in Chat Completions shape (`function_call_to_openai_chat` / `function_response_to_openai_chat`): tool calls nested under `tool_calls[].function` on the assistant message, tool results sent as standalone `role: "tool"` messages
  - `DeepSeekAdapter` adds DeepSeek V4 thinking-mode support (`deepseek-v4-flash` / `deepseek-v4-pro`, both 1M context):
    - Two YAML parameters drive it — `thinking_mode` (enum `enabled`/`disabled`, `request_key: thinking.type`, matching the API default `enabled`) and `reasoning_effort` (enum `low`/`high`/`max`, default `high`). Both are normalizer-mapped; only the `extra_body` relocation below is adapter code
    - `reasoning_effort` is a regular OpenAI SDK argument and is forwarded as-is. The `thinking` object is *not* part of the SDK's `chat.completions.create` signature, so `_build_request_params` moves it into `extra_body` (as DeepSeek's own SDK guidance prescribes)
    - Thinking mode ignores `temperature`, `top_p`, `presence_penalty`, and `frequency_penalty` (the API accepts them silently but they have no effect), so the adapter drops them whenever thinking is not disabled
    - The chain of thought comes back in `reasoning_content` and is captured as a `ThinkingResponse` by the shared base
    - Covered by `tests/test_deepseek_adapter.py`
  - DeepSeek and OpenRouter opt into the shared local-tool loop through `SUPPORTS_TOOLS`; only registered models declaring `function_calling` accept tools. Both plain callables and `BaseTool` instances are supported. Assistant calls are recorded before execution, results are appended individually with matching IDs, and exact native reasoning/tool data survives subsequent rounds and JSON persistence. Forced choices relax to `auto` after the first exchange. Terminal error/truncation/refusal states never execute tools. Rounds are bounded by 40 and an elapsed deadline; native async requests await coroutine tools and offload synchronous tools. Unknown tools, malformed arguments, and execution failures raise while preserving recorded history. OrcaRouter still rejects tool requests.
  - DeepSeek's three registered V4 models expose tools, tool choice, and JSON output. `structured_output` sets `response_format={type: "json_object"}` and adds a JSON instruction/example; supplied schemas guide the prompt only. DeepSeek does not enforce a strict schema, and the adapter returns JSON text without local schema validation. The existing Z.AI JSON mode has the same schema-guidance limitation.
  - OpenRouter exposes tools and JSON/schema output specifically for the registered Muse Spark 1.3 route. Requests using either feature set `extra_body.provider.require_parameters=True`, so routing must honor the requested parameters. This is a checked registry snapshot, not runtime model discovery or a claim that every OpenRouter route supports the same features.
  - `OpenRouterAdapter` sends the unified OpenRouter `reasoning` object through the OpenAI SDK's `extra_body`. The registered `meta/muse-spark-1.3` model (1M context, text/image) has mandatory reasoning, so its YAML schema exposes only the effort level (`minimal`/`low`/`medium`/`high`/`xhigh`, default `medium`) mapped to `reasoning.effort`.
  - `OpenAICompatibleAdapter` provides a native async path: `request_llm_async` mirrors the sync chat flow on a lazily constructed `AsyncOpenAI` client (`_build_async_client` / `async_client`), inherited as-is by `DeepSeekAdapter`, `OpenRouterAdapter`, and `OrcarouterAdapter`. `ZaiAdapter` explicitly pins `request_llm_async` back to the thread-offloaded `AdapterBase` default: the base's async path is backed by `AsyncOpenAI` (not the official `ZaiClient`), so inheriting it would bypass Z.AI-specific request handling
  - `OrcarouterAdapter` targets `https://api.orcarouter.ai/v1` with `ORCAROUTER_API_KEY`. No model currently routes to this adapter in the registry; its existing catalog-registration test is out of sync with the YAML
  - `ZaiAdapter` adds preserved thinking, tool calling, structured output, and web search on top of the shared base:
    - GLM-5.3 remains text-only and exposes forced thinking (`enabled`), effort `low`/`high`/`max` (default `max`), and JSON output. GLM-5.3-Flash exposes images, videos, and native PDF input. Image/video parts use base64 data URLs; PDFs use `file.file_data` plus `filename`. Native PDFs cannot be combined with image/video parts in the same message under the provider contract; this raises explicitly. Other documents keep text extraction, as do PDFs on the text-only model.
    - `glm-5.3-flash` exposes its forced-thinking contract through YAML: `thinking_mode` is fixed to `enabled` and mapped to `thinking.type`, `clear_thinking` defaults to `false` as recommended for coding/agent tasks, and `reasoning_effort` offers `low`/`high`/`max` with the existing platform default `high`. The adapter captures `reasoning_content` on every response and replays it verbatim in assistant history, including intermediate tool rounds, so preserved/interleaved thinking remains coherent.
    - **Function calling**: `request_llm` routes to a recursive `request_llm_with_functions` loop (request → execute local `BaseTool`/callable tools → append `FunctionCall`/`FunctionResponse` plus the assistant thinking block → re-ask) until the model stops emitting `tool_calls`. Function tools are emitted as `{"type": "function", "function": {...}}` via `_convert_function_to_tool` (reusing `BaseTool.to_params(provider="openai")` or `_callable_to_json_schema`)
    - **Structured output**: `structured_output` enables Z.AI JSON mode through `response_format: {type: "json_object"}` on both plain and tool-enabled requests. When the caller supplies a Pydantic model class or JSON Schema dict, the adapter adds that schema to the system instruction, following Z.AI's documented JSON-mode pattern (the API accepts `json_object`, not an embedded strict schema).
    - **Web search**: Z.AI's built-in server-side `web_search` tool, enabled by the `web_search` additional parameter. `_build_request_params` is overridden to attach the built-in tool (`{"type": "web_search", "web_search": {"enable": True, "search_engine": "search-prime", "search_result": True}}`) so both the plain-chat and function-calling paths pick it up. No static `search_query` is sent — GLM derives queries from the conversation. Built-in and function tools are merged on the same request

- `WiroAIAdapter`
  - Uses WiroAI's JSON HTTP API at `https://api.wiro.ai/v1`: it submits `POST /Run/{owner}/{model}` once, then polls the returned task through `POST /Task/Detail` with exponential backoff until a terminal status. The adapter never resubmits a generation while polling.
  - Supports API-key-only projects through `WIRO_API_KEY` and signature-auth projects when `WIRO_API_SECRET` is also set. Signature headers use WiroAI's documented HMAC-SHA256 scheme. `WIRO_API_BASE_URL` optionally overrides the endpoint.
  - Registers the text model `Qwen3.8-27B-Uncensored`, routed to WiroAI's live `Qwen/Qwen3.8-27B-Uncensored` slug. Provider route metadata lives in `models_config.yaml` (`wiro_owner` / `wiro_model`) so additional WiroAI models can reuse the adapter.
  - Sends the system prompt through WiroAI's `system_prompt` parameter. The platform-owned conversation is serialized into each `prompt`, avoiding hidden provider session state when a conversation is cleared, restored, or switched between adapters. Document attachments are included as extracted text; media files are rejected explicitly because this catalog entry is text-only.
  - Parses the final raw LLM output into answer text and separate `ThinkingResponse` records. WiroAI does not report token counts, so token usage fields are zero; the task's actual `totalcost` is preserved as `usage.costs`.
  - Uses the `AdapterBase` thread-offloaded async fallback and does not support tool calling.

- `KimiAdapter`
  - Uses Moonshot AI's OpenAI-compatible Chat Completions endpoint at `https://api.moonshot.ai/v1` through the existing `openai` dependency and `MOONSHOT_API_KEY`
  - Registers `kimi-k3` with its 1,048,576-token context window, text/image/video input, cached/uncached input pricing, and output pricing
  - Maps the platform's `max_tokens` setting to Kimi's `max_completion_tokens`; K3's fixed sampling parameters (`temperature`, `top_p`, `n`, `presence_penalty`, and `frequency_penalty`) are deliberately omitted
  - `tool_choice="required"` and named forced choices apply to the first request only; after a local tool exchange the adapter switches to `auto` so the model can produce a final answer. Caller parameter dictionaries are not changed.
  - Captures K3's `reasoning_content` as `ThinkingResponse` and reconstructs it with the complete assistant message on later turns, as required for multi-turn reasoning and tool calls
  - Supports local base64 image/video parts, text extraction for document files, strict JSON Schema structured output, and `reasoning_effort` choices `low`/`high`/`max` (unchanged default `max`)
  - Supports bounded sync and native async custom-function loops with `tool_choice` (`auto`, `none`, or `required`); tool-call assistant messages, reasoning, and matching tool results are preserved in history
  - K3 context caching is automatic on the provider side and requires no adapter-managed cache identifier

### Capability audit alignment (2026-09-08)

Grok 4.6 advertises a 500,000-token context window. Its output-token default remains separate and unchanged.

Gemini PDFs are limited to 50,000,000 bytes per file and 1,000 pages, including an aggregate 1,000-page check for locally assembled input. Exceeding these limits raises `ValueError` instead of silently dropping visual content through text extraction. Before each Interactions request (including full-history continuation retries), the adapter measures the JSON-encoded payload, including base64 expansion, prompts, and other attachments. It uploads the largest inline PDFs until the request fits a conservative 19 MB budget under the 20 MB inline limit, then sends `document.uri`. Sync and native async uploads use the corresponding Files API. Original local bytes and conversation history are unchanged; full replay reuploads as needed rather than persisting expiring file URLs. Upload errors propagate; files remain under provider retention so stored interactions can use them. This does not add upload support for oversized audio/video or count PDFs hidden in remote continuation state.

The contracts were checked against [GLM-5.3](https://docs.z.ai/guides/llm/glm-5.3), [Z.AI multimodal request schemas](https://docs.z.ai/api-reference/llm/chat-completion), [Kimi K3](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart), [Grok 4.6](https://docs.x.ai/developers/models/grok-4.6), [Mistral structured output](https://docs.mistral.ai/studio/conversations/structured-output/custom), [DeepSeek JSON mode](https://api-docs.deepseek.com/guides/json_mode/), and [OpenRouter's Muse Spark endpoint catalog](https://openrouter.ai/api/v1/models/meta/muse-spark-1.3/endpoints). Gemini's [PDF limits](https://ai.google.dev/gemini-api/docs/document-processing#technical-details) and Python Files API were read through `geminiDocs` in `.mcp.json`; Interactions document URI and Mistral response-format payloads were also checked against installed SDK types.

`tests/test_provider_capabilities.py` covers registry exposure, JSON output, sync/async tool execution and reasoning replay, terminal guards, round limits, visual serialization, and PDF boundary/upload behavior with mocked services. These are offline regressions; no paid generation calls were made.

### 7.2 Async support
`AdapterBase` provides a default `request_llm_async` that runs the adapter's synchronous `request_llm` off the event loop via `asyncio.to_thread`. As a result `APIHandler.request_async` / `request_llm_async` work for every adapter rather than only OpenAI.

`OpenAIAdapter` overrides the default with a native async implementation (`request_llm_async`, `request_llm_with_functions_async`) backed by the async OpenAI client. `AnthropicAdapter` likewise overrides it with a native implementation backed by `anthropic.AsyncAnthropic` (covering the simple, streaming, and tool-use paths, including async token counting). `GoogleAdapter` overrides it with a native implementation on the google-genai async surface `client.aio` (covering the standard Interactions chat/tool loop, Deep Research, and Antigravity paths, with async polling). `OpenAICompatibleAdapter` overrides it with a native `AsyncOpenAI`-backed chat path, giving `DeepSeekAdapter`, `OpenRouterAdapter`, and `OrcarouterAdapter` native async for free. The remaining adapters (Grok, Mistral, WiroAI, and Z.AI — which deliberately pins itself back to the default, see §7.1) use the thread-offloaded fallback.

`KimiAdapter` also uses a native `AsyncOpenAI` client and adds an async tool loop that awaits coroutine tools directly. `WiroAIAdapter` uses the thread-offloaded fallback because its implementation is backed by synchronous `requests` polling.

## 8. Multimodal behavior by adapter (implemented)
- OpenAI: text, image, audio, document inputs
- Anthropic: text/image/document
- Google: text/image/audio/document/video inputs; Gemini Deep Research agents through background Interactions API calls; Antigravity managed agent (`agent_type: antigravity`) over the Interactions API with a remote sandbox, accepting text/image input only
- Grok: text/image/document in chat
- Mistral: text/image/document chat
- DeepSeek/OpenRouter/OrcaRouter: text + image/document conversion (OpenAI-compatible payload)
- Z.AI: GLM-5.3 is text-only; GLM-5.3-Flash adds image/video/native PDF input via `zai-sdk` `ZaiClient`. Both support preserved thinking, function calling, structured JSON output, and the built-in `web_search` tool
- Kimi (Kimi K3): text/image/video input (base64 data URLs for local visual media), extracted document text, structured output, reasoning preservation, and custom function calling
- WiroAI (Qwen3.8-27B-Uncensored): text input plus extracted document text; structured answer/thinking output; no media or tool calling

## 9. Tools subsystem
Files: `tools/base.py` and concrete tools in `tools/*.py`

### 9.1 Base abstraction
`BaseTool` requires:
- callable interface (`__call__(...)`)
- nested `InputModel` Pydantic schema

`BaseTool.to_params(provider=...)` emits provider-specific tool declarations for:
- OpenAI
- Anthropic
- Google (JSON Schema preserved; `GoogleAdapter` wraps the declaration with `{"type": "function", ...}` for Interactions)
- Grok (the `xai_sdk` `tool` type is imported lazily inside `to_params`, so importing the tools layer does not require `xai_sdk`)

Plain Python callables passed as tools are converted by `AdapterBase._callable_to_json_schema` using Pydantic `TypeAdapter`; each adapter wraps that canonical schema in its provider-specific envelope. Pydantic resolves postponed annotations and emits nested models, recursive `$defs`/`$ref`, nullable unions, `Literal`/enum values, annotated constraints, `TypedDict`, and typed lists, dictionaries, tuples, and sets. Unannotated values remain unconstrained rather than being mislabeled as strings. Bound methods, partials, and callable instances use their effective signatures and stable tool names. Positional-only and variadic signatures, unresolved annotations, and types without a JSON Schema fail explicitly before a provider call.

`BaseTool.clean_schema` copies the schema and removes only schema titles and legacy boolean `required` extras emitted by `Field(required=True)`. It visits schema positions rather than arbitrary dictionaries, preserving actual nested `required` lists, reference siblings, constraints, and literal values in defaults/examples/enums, including fields and definitions named `title` or `required`. `resolve_schema_for_google` remains as a compatibility helper returning a detached copy; it no longer expands references or discards nullable branches and dictionary/numeric constraints.

OpenAI callable and `BaseTool` declarations explicitly use `strict: false`; Mistral callable declarations also disable strict mode. This preserves omission and Python default semantics without forcing optional properties to become required, introducing synthetic nulls, or closing typed dictionaries. A nullable annotation with no default is still required; omitting a defaulted argument leaves Python to supply the default, while an explicit JSON null remains `None`. Tool execution still passes decoded keyword arguments to the callable; schema generation does not instantiate nested Pydantic objects or add runtime validation. Provider-specific JSON Schema limits can still produce provider errors; constraints are never silently removed to force acceptance.

`tests/test_tool_schemas.py` checks accepted/rejected JSON instances, shared provider envelopes, recursion, defaults, nullability, typed containers, non-mutation, invalid signatures, and real sync/async OpenAI and Gemini SDK request serialization with network calls intercepted. The SDK checks were run with OpenAI 3.8.0 and google-genai 2.22.0; they do not establish live provider acceptance. Install offline test dependencies with `python -m pip install -r requirements-test.txt`.

Contracts checked through the MCP servers in `.mcp.json`: [OpenAI strict-mode opt-out](https://developers.openai.com/api/docs/guides/function-calling#strict-mode) and [Gemini structured outputs, recursive schemas, and named response format](https://ai.google.dev/gemini-api/docs/structured-output).

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
- `ORCAROUTER_API_KEY`
- `WIRO_API_KEY`
- `WIRO_API_SECRET` (optional; signature-auth projects)
- `WIRO_API_BASE_URL` (optional endpoint override)
- `MISTRAL_API_KEY`
- `ZAI_API_KEY`
- `MOONSHOT_API_KEY`

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
1. Tool-calling support is partial across adapters (implemented in OpenAI/Anthropic/Google/Grok/Mistral/Z.AI/Kimi/DeepSeek/OpenRouter, not in OrcaRouter/WiroAI).
2. Older, unversioned conversation files cannot recover IDs or native response data that were never saved. They load with safe defaults and rebuild history without guessing provider ownership from IDs.
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
3. Adapter checks the response status before executing local tools. Failed, incomplete, refused, cancelled, pending, and unrecognized results return control with their partial output and metadata. Grok hosted calls are excluded from local execution.
4. Tool outputs are captured as `FunctionResponse` records.
5. Conversation is updated with tool call/response records.
6. Adapter calls the provider again until a final result or a status requiring caller attention is returned, bounded by `MAX_TOOL_ROUNDS` (40); exceeding the bound raises `RuntimeError`. Claude also continues hosted pauses in this loop. Kimi, DeepSeek, and OpenRouter relax the initial forced tool choice to `auto`.

### 14.3 Gemini Deep Research flow
1. Client calls a model such as `deep-research-preview-04-2026` or `deep-research-max-preview-04-2026`.
2. `models_config.yaml` marks the model with `background_mode: true` and `agent_type: deep_research`, so `GoogleAdapter` uses the Gemini Interactions API instead of `models.generate_content`.
3. The full history (first call) or all messages after a valid provider/model checkpoint is converted into Interactions steps. The conversation system prompt is included as an inline instruction step rather than `system_instruction`.
4. Images, audio, video, and small PDFs use inline base64 content; PDFs are uploaded through the Files API when needed to fit the encoded request budget. Office/text documents are converted to text content.
5. The adapter starts the interaction with `agent=<model>`, `background=True`, and `store=True`; it does not send `generation_config` because Gemini agents require agent-specific configuration through `agent_config`.
6. The adapter polls while the interaction is `queued` or `in_progress`, within the shared turn deadline. A terminal or action-required status returns control; unfamiliar statuses are exposed as `unknown` rather than polled indefinitely.
7. The returned interaction is checkpointed for later turns and its complete native steps are retained for replay. For display, the adapter uses the May 2026 steps schema: the adapter walks `interaction.steps`, picks `model_output` steps, and pulls text / image / annotation items out of each step's `content[]` array. Text content joins into the assistant message body, image content becomes `ImageFile` attachments, citation annotations become structured `citations` (legacy formatted `additional_responses` remain available), hosted steps become `hosted_tool_results`, and `interaction.usage` (`total_input_tokens`, `total_output_tokens`, `total_tokens`) is mapped to the usual usage keys.

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

## 16. Self-improving agents: Archon pipeline
File: `self_improving_agents/archon.py`

An implementation of the Archon inference-time architecture ("Archon: An Architecture Search Framework for Inference-Time Techniques", PDF in `docs/`). Instead of one LLM call, a prompt flows through a pipeline of LLM components that generate, critique, filter, and merge candidate answers. Every LLM call goes through the platform facade (`APIHandler`), so any model in `models_config.yaml` can play any role.

- `Archon(generator_models, fuser_model, critic_model=None, ranker_model=None, verifier_model=None, unit_test_model=None, samples_per_generator=1, top_k=3, num_unit_tests=5, system_prompt=...)` — pipeline configuration; each optional component is enabled by supplying a model name for it.
- `generate(prompt, files=None, generator_parameters=None) -> ArchonResult` (sync wrapper over `asyncio.run`) and `generate_async(...)` for callers already inside an event loop.
- Pipeline order follows the paper's construction rules: **Generators** (first layer, run in parallel via `request_async`) → **Critic** (one call producing strengths/weaknesses per candidate) → **Ranker** (orders candidates, keeps `top_k`) → **Verifier** (two-stage reasoning + `[Correct]`/`[Incorrect]` verdict per candidate, run in parallel) → **Unit Test Generator/Evaluator** (writes assertions from the prompt, scores each candidate, keeps the best-scoring ones) → a single final **Fuser** (always the last layer; uses the with-critiques prompt variant when critiques exist).
- Component prompts are module constants transcribed from Tables 10–23 of the paper; the verifier stage-2 verdict prompt follows the two-stage procedure described in the paper's Section 3.1.
- Each call uses a fresh `APIHandler` so component calls are isolated conversations; `files` (e.g. `ImageFile`) are passed to every component so critics/rankers/fusers can judge candidates against the attachment; `generator_parameters` are normalized per model by the facade and applied to generator calls only.
- Robustness behavior: failed generators are logged and skipped (error only if all fail); unparseable ranker output falls back to original order; an unparseable verifier verdict fails open (candidate kept); if the verifier rejects everything, all candidates are kept; each dropped candidate records `dropped_by` (`ranker`/`verifier`/`unit_tests`) in the returned `ArchonResult.candidates` trace.
- Imported as a regular subpackage: `from llm_platform.self_improving_agents.archon import Archon`.

## 17. File and package map
- `core/llm_handler.py`: orchestration facade (lazy adapter registry, conversation state, sync/async routing)
- `core/parameter_normalizer.py`: `ParameterNormalizer` parameter pipeline
- `helpers/model_config.py`: YAML model registry (cached + name-indexed) and parameter normalization
- `services/conversation.py`: provider-agnostic conversation and tool metadata classes (no vendor knowledge); platform-internal persistence only (`save_to_json`/`read_from_json`)
- `adapters/serializers.py`: provider wire (de)serialization for the domain objects (functions like `function_call_to_openai` for the OpenAI Responses API and `function_call_to_openai_chat` for the OpenAI-compatible Chat Completions API), kept out of the domain model so `services/` stays provider-agnostic
- `services/files.py`: file classes and text/media extraction
- `adapters/adapter_base.py`: `AdapterBase` contract + shared adapter helpers
- `adapters/openai_compatible_adapter.py`: `OpenAICompatibleAdapter` base (DeepSeek, OpenRouter, OrcaRouter, Z.AI, Kimi)
- `adapters/orcarouter_adapter.py`: `OrcarouterAdapter` — OrcaRouter models through the OpenAI-compatible endpoint using `ORCAROUTER_API_KEY`
- `adapters/wiro_ai_adapter.py`: `WiroAIAdapter` — WiroAI Run/Task submission, polling, authentication, and structured LLM output parsing
- `adapters/zai_adapter.py`: `ZaiAdapter` — Z.AI GLM models via the official `zai-sdk` `ZaiClient` (OpenAI-compatible); adds preserved/interleaved thinking, function calling (recursive tool loop), structured JSON output, and the built-in `web_search` tool
- `adapters/kimi_adapter.py`: `KimiAdapter` — Kimi models through Moonshot AI's OpenAI-compatible endpoint; adds K3 parameter rules, reasoning preservation, image/video input, structured output, and sync/async function calling
- `adapters/*.py`: provider integrations
- `self_improving_agents/archon.py`: `Archon` inference-time pipeline (generate → critique → rank → verify → unit-test → fuse) built on `APIHandler` (see §16)
- `tools/base.py`: `BaseTool` contract + per-provider declaration emission
- `tools/ssh_command.py`: `SSHCommandTool` base for SSH admin tools
- `tools/*.py`: callable tool implementations
- `models_config.yaml`: model routing and metadata
- `types.py`: typed `AdditionalParameters`
