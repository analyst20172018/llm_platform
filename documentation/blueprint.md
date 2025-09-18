# LLM Platform Blueprint

## 1. Product Context
- **Mission**: Deliver a single Python interface that shields application teams from provider-specific APIs while unlocking advanced multimodal, tool-calling, and voice capabilities.
- **Strategic Value**: Accelerates product experiments, reduces vendor lock-in, and centralises governance for model usage, cost, and compliance.
- **Guiding Principles**: Provider-agnostic design, composable architecture, thoughtful defaults with escape hatches, and observability baked into each layer.

## 2. Target Users & Primary Use Cases
- **Application Engineers** – build end-user LLM features without juggling multiple SDKs.
- **Platform / ML Engineers** – integrate new providers, manage configuration, enforce compliance guardrails.
- **Ops / Support Analysts** – audit conversations, review model usage, tune cost/performance mix.
- **Automation Developers** – orchestrate tools (PowerShell, Raspberry Pi, Ubuntu admin scripts, etc.) via natural-language requests.

**Key Use Cases**
1. Unified chat completion API with automatic conversation history management.
2. Switching models mid-conversation while retaining state and tool context.
3. Multimodal prompts that blend text, PDFs, spreadsheets, images, audio.
4. Function/tool calling with custom Python or remote executors.
5. Voice workflows (Speechmatics/OpenAI transcription, ElevenLabs speech synthesis).
6. Image generation across OpenAI and Google providers.

## 3. Functional Scope
- **Core Conversation Flow**: `core.llm_handler.APIHandler` mediates between clients and adapters, persisting messages via `services.conversation` abstractions.
- **Adapter Abstraction**: Every provider extends `adapters.adapter_base.AdapterBase`, normalising request/response payloads, tool semantics, and async variants.
- **File & Modal Support**: `services.files` family loads and normalises text, PDFs, Excel, images, audio, and video artifacts; base64 conversion handled internally.
- **Model Catalog**: `helpers.model_config.ModelConfig` hydrates model metadata from `models_config.yaml`, exposing pricing, modality, and feature switches.
- **Tooling Surface**: `tools` package offers reusable automation primitives and Pydantic-backed schemas for function calling.
- **Extensibility**: New providers drop-in by adding an adapter, configuring models, and wiring credentials through `.env` keys described in `README.md`.

## 4. Non-Goals (Current)
- End-user UI, chat frontend, or hosted API gateway.
- Automatic billing reconciliation across vendors.
- Persistent vector storage, RAG orchestration, or workflow DAG editing.

## 5. High-Level Architecture
```
+--------------+        +-----------------+        +-------------------+
| Client Code  | -----> | core.APIHandler | -----> | adapters.*        |
+--------------+        +-----------------+        |  (provider SDKs)  |
        |                        |                 +-------------------+
        |                        v                         |
        |                services.conversation             v
        |                        |                 External Provider APIs
        v                        v
 services.files         helpers.model_config
        |
        v
      tools
```
- **Boundary Management**: API handler is the façade; adapters encapsulate vendor-specific differences.
- **State**: Conversations and tool responses stored in-memory (extendable to persistent stores).
- **Configuration**: YAML-driven, enabling environment-based overrides.

## 6. Module Breakdown
- `core/`: Entry point (`llm_handler.py`) orchestrating requests, lazy adapter init, async pathways, token counting (via `tiktoken`).
- `adapters/`: Providers (OpenAI, Anthropic, Google, Grok, DeepSeek, OpenRouter, Speechmatics, ElevenLabs, Mistral) plus legacy OpenAI handler.
- `services/`: Conversation state machines (`Conversation`, `Message`, `FunctionCall`, `FunctionResponse`) and file abstractions for multimodal payloads.
- `helpers/`: Model configuration loader and accessors.
- `tools/`: Base Pydantic schema tooling plus concrete automation helpers (PowerShell, OS admins, Reddit, Raspberry Pi, Czech law domain tools).
- `docs/`: Deep dives per major subsystem to supplement this blueprint.

## 7. Configuration & Secrets
- **Model Catalog**: `models_config.yaml` captures pricing, capabilities flags (`function_calling`, `reasoning_effort`, `web_search`, etc.) and visibility toggles.
- **Environment Variables**: `.env` keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`, `GROK_API_KEY`, `OPENROUTER_API_KEY`, `SPEECHMATICS_API_KEY`, `ELEVENLABS_API_KEY`) required per provider.
- **Override Strategy**: Allow per-environment YAML overrides or dynamic registry injection for automated deployments.

## 8. External Dependencies
- **SDKs**: `openai`, `anthropic`, `google-genai`, `mistralai`, `xai_sdk`, `elevenlabs`, `speechmatics-python`.
- **Utilities**: `tiktoken` for token accounting, `pydantic` for schemas, `PyPDF2`/`pandas`/`pillow`/`pydub` for multimodal processing, `requests` for remote fetches, `praw` for Reddit tool.
- **Runtime**: Python 3.10+ recommended to align with async features and type hints.

## 9. Request Lifecycle
1. Client instantiates `APIHandler` with optional system prompt.
2. `request()` adds the user message (and optional files) to `Conversation`.
3. Handler resolves target adapter via `ModelConfig.get_adapter_name()`.
4. Adapter converts conversation history into provider format, submits SDK call.
5. Responses converted back into `Message`, with tool calls resolved recursively when provided.
6. Token usage and tool outputs persisted for follow-up requests.
