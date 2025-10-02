# LLM Platform

> A provider-agnostic Python toolkit for orchestrating multimodal, tool-aware LLM workflows behind a single interface.

## Table of Contents
- [Overview](#overview)
- [Feature Highlights](#feature-highlights)
- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [Getting Started](#getting-started)
- [Quickstart](#quickstart)
- [Core Concepts](#core-concepts)
- [Working with Files](#working-with-files)
- [Function and Tool Calling](#function-and-tool-calling)
- [Speech and Audio](#speech-and-audio)
- [Image Generation](#image-generation)
- [Model Configuration](#model-configuration)
- [Extending the Platform](#extending-the-platform)
- [Development Workflow](#development-workflow)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Overview
LLM Platform streamlines application development across large language model providers. It wraps the quirks of each vendor SDK (OpenAI, Anthropic, Google Gemini, DeepSeek, Grok, OpenRouter, Speechmatics, ElevenLabs, Mistral, and more) behind a single Python surface. The platform keeps conversation state, negotiates function calls, normalises multimodal payloads, and exposes thoughtful defaults with escape hatches when direct SDK access is required.

Use it to:
- Prototype and ship LLM-backed product features without committing to a single provider.
- Mix text, structured data, images, audio, and tool execution inside one conversation loop.
- Maintain observability over model usage, token consumption, and tool outputs.
- Onboard new providers quickly by implementing a thin adapter.

## Feature Highlights
- **Unified API handler** – One entry point (`APIHandler`) for synchronous and asynchronous requests across vendors.
- **Stateful conversations** – Built-in conversation objects persist history, usage metrics, and tool exchange metadata.
- **Function/tool calling** – Invoke Python callables or custom `BaseTool` implementations from any provider that supports tool use.
- **Multimodal requests** – Attach text, PDFs, spreadsheets, images, audio, or video in a single prompt.
- **Voice workflows** – Convert speech to text via OpenAI or Speechmatics and reuse the transcripts immediately.
- **Image generation** – Produce images with OpenAI or Google using the same interface.
- **Model catalog** – YAML-driven registry describes pricing, capabilities, and adapter routing for every model.
- **Provider extensibility** – Drop in new adapters without touching the core request flow.

## Architecture
At runtime the platform composes a façade (`core.APIHandler`) with provider-specific adapters and shared services:

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

- **core/** – Request orchestration, conversation bookkeeping, token counting, synchronous/async entry points.
- **adapters/** – Provider shims translating between platform message formats and vendor SDKs.
- **services/** – Conversation models plus file ingestion layers that unify text, document, and media handling.
- **helpers/** – Model catalog loader and helper utilities.
- **tools/** – `BaseTool` abstractions and off-the-shelf automations callable from any conversation.

## Repository Layout
```
llm_platform/
├── core/                 # API handler and request orchestration
├── adapters/             # Provider integrations (OpenAI, Anthropic, Google, ...)
├── services/             # Conversation state, message & file primitives
├── tools/                # Built-in automation helpers and schemas
├── helpers/              # ModelConfig and shared helpers
├── docs/                 # Deep dives on subsystems
├── documentation/        # Blueprint and architecture reference material
├── models_config.yaml    # Canonical model registry
├── requirements.txt      # Runtime dependencies
└── README.md             # You are here
```

## Getting Started
### Prerequisites
- Python 3.10+
- Virtual environment recommended (`python -m venv .venv && source .venv/bin/activate`)
- Provider credentials for the services you intend to call

### Installation
```bash
git clone https://github.com/analyst20172018/llm_platform
cd llm_platform
pip install -r requirements.txt
```

### Environment Variables
Populate a `.env` file (or export variables in your shell) with the provider credentials you plan to use:
```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=...
GROK_API_KEY=...
OPENROUTER_API_KEY=...
SPEECHMATICS_API_KEY=...
ELEVENLABS_API_KEY=...
MISTRAL_API_KEY=...
```
Only the keys for providers you call are required at runtime.

## Quickstart
```python
from llm_platform.core.llm_handler import APIHandler

handler = APIHandler(system_prompt="You are a helpful assistant.")

result = handler.request(
    model="gpt-4o",
    prompt="Summarise the key takeaways from the latest quarterly report in three bullet points.",
    temperature=0.2,
)

print(result.content)
```

### Switching models mid-conversation
```python
handler.request(
    model="claude-3-5-sonnet-latest",
    prompt="Great. Now draft a follow-up email in a confident tone.",
)

handler.request(
    model="gemini-2.0-pro",
    prompt="Translate that email into Spanish and cite your sources if you used any.",
    additional_parameters={"citations": True}
)
```
Conversation history is preserved and normalised automatically, even as providers change.

## Core Concepts
- **`APIHandler`** – Central façade. Handles request routing, adapter initialisation, and token accounting. Supports sync (`request`) and async (`request_async`) flows.
- **`Conversation`** – Stores message history, usage statistics, and tool exchanges. Automatically updated after each request.
- **`Message`** – Represents user or assistant turns (text, tool calls, structured metadata) within a conversation.
- **`BaseFile` & subclasses** – Canonical file wrappers (`TextDocumentFile`, `PDFDocumentFile`, `ExcelDocumentFile`, `ImageFile`, `AudioFile`, `VideoFile`) used to send multimodal inputs.
- **`BaseTool`** – Pydantic-backed abstraction for function calling across providers. Return values propagate back into the conversation.
- **`ModelConfig`** – Loads `models_config.yaml` to expose provider metadata, default parameters, pricing hints, and adapter routing.

## Working with Files
Attach files using the high-level constructors in `llm_platform.services.files`:
```python
from pathlib import Path
from llm_platform.services.files import PDFDocumentFile, ImageFile

report = PDFDocumentFile.from_path(Path("reports/q4.pdf"))
diagram = ImageFile.from_path(Path("assets/architecture.png"))

reply = handler.request(
    model="gpt-4o",
    prompt="Summarise the PDF and describe the architecture diagram.",
    files=[report, diagram],
)

print(reply.content)
for attachment in reply.files:
    print(f"Attachment returned: {attachment.name} ({attachment.extension})")
```
Files are automatically converted to the provider-specific payload format (base64, binary uploads, etc.) by the active adapter.

## Function and Tool Calling
```python
from llm_platform.tools.base import BaseTool
from pydantic import BaseModel, Field

class WeatherTool(BaseTool):
    class InputModel(BaseModel):
        city: str = Field(description="City name")
        units: str = Field(description="Measurement system", default="metric")

    def __call__(self, city: str, units: str = "metric") -> dict:
        forecast = fetch_forecast(city, units)  # your implementation
        return {"forecast": forecast}

response = handler.request(
    model="claude-3-5-sonnet-latest",
    prompt="Check the weather in Berlin and let me know if I need an umbrella.",
    functions=[WeatherTool()],
)
```
Adapters translate tool schemas and invocations for each provider (OpenAI tool calls, Anthropic tool use, Gemini function calling, etc.). Tool outputs are appended to the conversation and optionally streamed through `tool_output_callback`.

## Speech and Audio
```python
with open("customer_call.mp3", "rb") as audio:
    transcript = handler.voice_to_text(audio, audio_format="mp3", provider="speechmatics")

follow_up = handler.request(
    model="gpt-4o",
    prompt=f"Summarise this call and flag any action items: {transcript}",
)
```
Use `voice_file_to_text(path)` for convenience when working with local files. Speech-to-text currently supports OpenAI and Speechmatics providers.

## Image Generation
```python
image_urls = handler.generate_image(
    prompt="A cyberpunk cityscape with neon reflections after rain",
    provider="openai",
    n=1,
    size="1024x1024",
)
```
Switch `provider="google"` to target Gemini image generation. Returned assets mirror the provider’s native format (URL, base64, or binary).

## Model Configuration
`models_config.yaml` contains the canonical registry of supported models. Each entry defines:
- Adapter name (routes to an adapter inside `adapters/`).
- Modalities and features (tool calling, vision, audio, web search, reasoning budget, etc.).
- Pricing and metadata useful for observability dashboards.

Override or extend the catalog by editing the YAML file or supplying environment-specific overlays. `ModelConfig` exposes helper methods to query models, resolve adapters, and fetch defaults.

## Extending the Platform
### Adding a New Provider
1. Create `adapters/<provider>_adapter.py` by subclassing `AdapterBase`.
2. Implement the required interface (`request`, `request_async`, optional helpers for images/voice).
3. Add your models to `models_config.yaml` with `adapter_name` pointing to the new adapter.
4. Document required environment variables in the README and `.env`.

### Adding Tools
- Derive from `BaseTool` and provide a Pydantic `InputModel`.
- Implement `__call__` or `__async_call__` to perform the action.
- Register the tool by passing instances into `APIHandler.request()` / `request_async()`.

### Custom Conversation Storage
`services.conversation.Conversation` stores messages in memory by default. To persist history elsewhere, subclass it and inject your implementation into `APIHandler` or wrap the handler with your own repository layer.

## Development Workflow
- Clone the repository and install dependencies inside a virtual environment.
- Run your preferred formatter/linters (the project targets idiomatic Python with type hints).
- Add or update tests (pytest is recommended; test suite layout forthcoming).
- Before committing, ensure `models_config.yaml` and any documentation reflect new adapters, tools, or configuration switches.

## Documentation
- `documentation/blueprint.md` – System roadmap and architectural reference.
- `docs/` – Deeper dives into subsystems (adapters, tools, governance, etc.).
- `AGENTS.md` – Guidance on conversational agent behaviours and templates.

## Contributing
Issues and pull requests are welcome. Please open a discussion for significant design changes to align on adapter interfaces, configuration format, and dependency impacts before implementing.

## License
License information forthcoming.
