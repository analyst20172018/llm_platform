# LLM Platform Technical Documentation

Version: 2026-02-20
Source of truth: current implementation in this repository (`core/`, `adapters/`, `services/`, `helpers/`, `tools/`, `models_config.yaml`)

## 1. Purpose and scope
This project is a provider-agnostic Python platform for:
- Multi-provider LLM text interactions through one facade (`APIHandler`)
- Tool/function calling during conversations
- Multimodal input handling (text, images, audio, PDF, Excel, Word, PowerPoint, video)
- Speech-to-text integrations
- Image and video generation integrations
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
- `adapters/*.py`: provider-specific translation and API calls
- `services/conversation.py`: conversation/message/function-call domain model + serialization
- `services/files.py`: file abstractions and format conversion/extraction
- `helpers/model_config.py`: YAML model registry loader and parameter schema normalization
- `tools/*.py`: tool abstraction and built-in tool implementations
- `types.py`: typed definition of supported additional parameters

## 3. Core facade (`APIHandler`)
File: `core/llm_handler.py`

### 3.1 Responsibilities
- Adapter registry and lazy adapter initialization
- Conversation state ownership (`self.the_conversation`)
- Parameter normalization (`_prepare_additional_parameters`)
- Sync and async request routing
- Convenience APIs for STT, image generation, video generation, and image editing

### 3.2 Public API
- `request(model, prompt, functions=None, files=[], tool_output_callback=None, additional_parameters=None, **kwargs) -> Message`
- `request_async(...) -> Message`
- `request_llm(...) -> Message` (internal/public dispatch, no user-message append)
- `request_llm_async(...) -> Message`
- `voice_to_text(audio_file, audio_format, provider='openai', **kwargs)`
- `voice_file_to_text(audio_file_name, provider='openai', **kwargs)`
- `generate_image(prompt, provider='openai', n=1, **kwargs)`
- `generate_video(prompt, provider='google', **kwargs)`
- `edit_image(prompt, provider='openai', images=..., n=1, **kwargs)`
- `get_models(adapter_name) -> List[str]`
- `calculate_tokens(text) -> {'bytes': int, 'tokens': int}` (tiktoken `cl100k_base`)

### 3.3 Adapter resolution
Adapter class is selected by model's `adapter` in `models_config.yaml` and instantiated on demand.

Registered adapter classes include:
- `OpenAIAdapter`, `OpenAIImageAdapter`, `OpenAIOldAdapter`
- `AnthropicAdapter`
- `GoogleAdapter`
- `GrokAdapter`, `GrokImageAdapter`
- `DeepSeekAdapter`
- `OpenRouterAdapter`
- `MistralAdapter`
- `SpeechmaticsAdapter`
- `ElevenLabsAdapter`
- `AssemblyAIAdapter`

### 3.4 Additional parameter normalization pipeline
Implemented in `_prepare_additional_parameters`:
1. Merge user `additional_parameters` and deprecated `**kwargs` (kwargs only fill missing keys).
2. Load model parameter definitions from YAML.
3. Apply defaults for definitions with `send_default: true` (including `max_tokens`, which is a standard YAML `additional_parameter` per model with provider-specific `request_key` mapping, e.g. `max_output_tokens` for OpenAI and Google).
4. Map friendly keys to nested request keys (`request_key`, e.g. `reasoning_effort -> reasoning.effort`, `max_tokens -> max_output_tokens`).
5. Drop fields where `include_in_request: false`.
6. Filter unsupported keys for the selected model and log warnings.

## 4. Conversation domain model
File: `services/conversation.py`

Classes:
- `Conversation`: message list + system prompt + usage aggregations + serialization
- `Message`: role/content/files/usage/thinking/tool calls/tool responses
- `FunctionCall`: normalized function invocation metadata
- `FunctionResponse`: normalized tool output, optional files parsing
- `ThinkingResponse`: normalized reasoning/thinking content

Message roles: `user`, `assistant`, `function`

### Serialization
- `Conversation.save_to_json()` serializes messages and file payloads
- `Conversation.read_from_json(data)` reconstructs conversation and supported file objects

## 5. File abstraction model
File: `services/files.py`

### 5.1 File typing
`define_file_type(file_name)` classifies: `text`, `pdf`, `excel`, `word`, `powerpoint`, `image`, `audio`, `video`, `unknown`.

### 5.2 Class hierarchy
- `BaseFile`
- `DocumentFile`
  - `TextDocumentFile`
  - `PDFDocumentFile`
  - `ExcelDocumentFile`
  - `WordDocumentFile`
  - `PowerPointDocumentFile`
- `MediaFile`
  - `ImageFile`
  - `AudioFile` (auto-converts non-mp3 input to mp3)
  - `VideoFile`

### 5.3 Content extraction behavior
- PDF: text extraction via `PyPDF2`
- Excel: sheet text extraction via `pandas`
- Word (`.docx`): OOXML XML extraction from zip
- PowerPoint (`.pptx`): OOXML slide text extraction from zip

## 6. Model registry and metadata
File: `helpers/model_config.py`, config in `models_config.yaml`

### 6.1 Current catalog summary
- Total models: 27
- Visible models: 20
- Adapter families: 13

Models are grouped by `adapter`, with metadata:
- `name`, `display_name`
- `inputs`, `outputs`
- `pricing`
- `max_tokens`, `context_window`
- `visible`
- `additional_parameters` schema
- optional `background_mode`

### 6.2 Parameter schema capabilities
`Model` normalizes `additional_parameters` and supports:
- type normalization (`string`, `enum`, `boolean`, etc.)
- default UI metadata (`ui`, `label`)
- option normalization (including ratio-like values)
- request mapping (`request_key`)
- flags: `send_default`, `include_in_request`

## 7. Adapter capability matrix

### 7.1 Text and tool orchestration adapters
- `OpenAIAdapter`
  - Responses API based chat flow
  - Sync + async request methods
  - Tool calling with recursive loop
  - Supports `web_search`, `code_execution`, structured output parsing, reasoning/text parameter pass-through
  - Supports file citations retrieval from container files
- `AnthropicAdapter`
  - Sync only
  - Non-streaming and streaming execution paths
  - Streaming auto-enabled for large `max_tokens` (>= 21000)
  - Recursive tool-use loop
  - Supports web search, code execution, reasoning controls, structured output (non-streaming)
  - Performs max-token correction against context window
- `GoogleAdapter`
  - Sync request loop with tool execution
  - Supports function calling for `BaseTool` tools
  - Supports `web_search`, `url_context`, `code_execution`, structured output schema shaping, reasoning config
- `GrokAdapter`
  - Sync chat with optional tool execution loop
  - Supports web search and code execution tools in xAI SDK
- `MistralAdapter`
  - Sync chat, recursive function-calling, OCR mode, and audio transcription mode
- `OpenAIOldAdapter`
  - Legacy chat completions path, including audio modalities and recursive function calling
- `DeepSeekAdapter`
  - OpenAI-compatible chat completion path
  - Basic text/image input conversion
  - No tool execution implementation (`NotImplemented`)
- `OpenRouterAdapter`
  - OpenAI-compatible chat completion path
  - Basic text/image input conversion
  - No tool execution implementation (`NotImplemented`)

### 7.2 Specialized adapters
- `OpenAIImageAdapter`
  - Dedicated OpenAI image model adapter
  - Generates or edits image based on presence of input image files
- `GrokImageAdapter`
  - Dedicated xAI image adapter
  - Generates or edits one image
- `SpeechmaticsAdapter`
  - Speechmatics batch transcription
- `ElevenLabsAdapter`
  - ElevenLabs Scribe transcription with optional diarization formatting
- `AssemblyAIAdapter`
  - AssemblyAI transcription with polling

### 7.3 Async support
Currently implemented async LLM paths:
- `APIHandler.request_async`
- `OpenAIAdapter.request_llm_async`
- `OpenAIAdapter.request_llm_with_functions_async`
- `OpenAIAdapter.generate_video`

Other adapters are sync-only from the `APIHandler` perspective.

## 8. Multimodal behavior by adapter (implemented)
- OpenAI: text, image, audio, document inputs; image generation/editing; video generation; STT
- Anthropic: text/image/document; no STT
- Google: text/image/audio/document/video inputs; image generation; video generation
- Grok: text/image/document in chat; image generation/editing (separate image adapter)
- Mistral: text/image/document chat + OCR + STT
- DeepSeek/OpenRouter: text + image/document conversion (OpenAI-compatible payload)
- Speechmatics/ElevenLabs/AssemblyAI: STT-only workflows

## 9. Tools subsystem
Files: `tools/base.py` and concrete tools in `tools/*.py`

### 9.1 Base abstraction
`BaseTool` requires:
- callable interface (`__call__(...)`)
- nested `InputModel` Pydantic schema

`BaseTool.to_params(provider=...)` emits provider-specific tool declarations for:
- OpenAI
- Anthropic
- Google (schema transformed to Gemini-compatible form)
- Grok

### 9.2 Built-in tool modules
- `RunPowerShellCommand` (persistent PowerShell process)
- `CzechLaws` + helper function variant
- `Reddit`
- `RaspberryAdmin` (SSH)
- `UbuntuAdmin` (SSH)

## 10. Environment variables and credentials
Current code expects:
- `OPENAI_API_KEY` (OpenAI SDK default)
- `ANTHROPIC_API_KEY`
- `GOOGLE_GEMINI_API_KEY`
- `XAI_API_KEY`
- `DEEPSEEK_API_KEY`
- `OPENROUTER_API_KEY`
- `MISTRAL_API_KEY`
- `SPEECHMATICS_API_KEY`
- `ELEVEN_API_KEY`
- `ASSEMBLYAI_API_KEY`

Important: current `README.md` and some older docs list different variable names for Google/Grok/ElevenLabs. The values above reflect actual adapter code.

## 11. Dependencies
From `requirements.txt`:
- Provider SDKs: `openai`, `anthropic`, `google-genai`, `mistralai`, `xai_sdk`, `elevenlabs`, `speechmatics-python`, `assemblyai`
- Data/media: `pandas`, `pillow`, `PyPDF2`, `pydub`, `lxml`
- Tooling and support: `pydantic`, `python-dotenv`, `PyYAML`, `requests`, `tiktoken`, `loguru`, `rich`, `praw`

## 12. Error handling and observability
- Logging uses `loguru` in orchestration and adapters.
- `APIHandler.request_llm` catches adapter exceptions and appends an assistant error message.
- Several adapters append user-readable error messages to conversation when input validation fails (especially STT adapters).
- Some adapter methods remain `NotImplemented` and will raise directly.

## 13. Known implementation gaps and inconsistencies
1. Env-var naming inconsistency between docs and implementation (`GOOGLE_GEMINI_API_KEY` vs `GOOGLE_API_KEY`, `XAI_API_KEY` vs `GROK_API_KEY`, `ELEVEN_API_KEY` vs `ELEVENLABS_API_KEY`).
2. Some `get_models()` methods are incomplete or missing `return` of `NotImplemented` in older adapters.
3. Tool-calling support is partial across adapters (fully implemented in OpenAI/Anthropic/Google/Grok/Mistral, not in DeepSeek/OpenRouter).
4. Mutable default arguments exist in `Message` initializer (`[]` defaults), which is a Python risk pattern.
5. Adapter base contract is not uniformly inherited by all specialized adapters (`OpenAIImageAdapter`, `GrokImageAdapter`, `ElevenLabsAdapter`, `AssemblyAIAdapter`).
6. Legacy and current OpenAI paths coexist (`OpenAIAdapter` and `OpenAIOldAdapter`).

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
6. Adapter recursively calls provider until final non-tool assistant output is produced.

### 14.3 Speech-to-text flow
1. User supplies one audio file in latest message.
2. STT adapter validates input and optional language/diarization parameters.
3. Adapter submits job/polls or performs direct transcription call.
4. Assistant message returns transcript text.

## 15. Extending the platform

### 15.1 Add a new model
1. Add entry to `models_config.yaml` with:
   - `name`, `adapter`, `inputs`, `outputs`, token/context limits
   - optional `additional_parameters` definitions
2. Ensure the mapped adapter exists in `APIHandler._lazy_initialization_of_adapter`.

### 15.2 Add a new adapter
1. Implement adapter class in `adapters/` (prefer inheriting `AdapterBase`).
2. Implement at least `request_llm` and conversation conversion.
3. Add adapter mapping in `APIHandler` lazy-init map.
4. Add model entries in `models_config.yaml`.

### 15.3 Add a new tool
1. Subclass `BaseTool`.
2. Provide Pydantic `InputModel`.
3. Implement `__call__`.
4. Pass tool instance in `APIHandler.request(..., functions=[...])`.

## 16. File and package map
- `core/llm_handler.py`: orchestration facade
- `helpers/model_config.py`: YAML model registry and parameter normalization
- `services/conversation.py`: conversation and tool metadata classes
- `services/files.py`: file classes and text/media extraction
- `adapters/*.py`: provider integrations
- `tools/*.py`: callable tool implementations
- `models_config.yaml`: model routing and metadata
- `types.py`: typed `AdditionalParameters`

