# Adapters Documentation

## Overview

The adapters module is a core component of the LLM Platform that provides a consistent interface to various language model providers. Each adapter implements a common interface defined in `adapter_base.py` while handling the specific requirements and features of different LLM providers.

## AdapterBase

`AdapterBase` is the abstract base class that defines the common interface for all LLM provider adapters.

### Key Methods

```python
def convert_conversation_history_to_adapter_format(self, the_conversation, model=None, **kwargs)
```
Transforms conversation history to provider-specific format.

```python
def request_llm(self, model, the_conversation, functions=None, temperature=0, tool_output_callback=None, **kwargs)
```
Main method to request text generation from an LLM.

```python
def request_llm_with_functions(self, model, the_conversation, functions, temperature=0, tool_output_callback=None, **kwargs)
```
Requests LLM completion with tool/function calling capabilities.

```python
def generate_image(self, prompt, n=1, **kwargs)
```
Creates images from text prompts.

### Async Versions

Most adapters also implement asynchronous versions of these methods for non-blocking operations.

## Provider-Specific Adapters

### AnthropicAdapter

Adapter for Anthropic's Claude models.

**Key Features:**
- Handles multiple content types (text, images, documents)
- Supports Claude's thinking/reasoning process
- Implements tool calling with function execution
- Handles document citations

**Provider-Specific Details:**
- Implements computer_use beta feature
- Manages token budget for reasoning
- Special handling for Claude 3.7 Sonnet beta features

### OpenAIAdapter

Adapter for OpenAI's models (GPT, o1, etc.).

**Key Features:**
- Handles multiple content types (text, images, audio)
- Full implementation of function calling
- Image generation via DALL-E 3
- Voice transcription via Whisper

**Provider-Specific Details:**
- Special handling for o1/o3 models (different system prompt format)
- Audio support for GPT-4o-audio models

### GoogleAdapter

Adapter for Google's Gemini models.

**Key Features:**
- Handles multiple content types (text, images, documents, audio)
- Function calling implementation
- Image generation

**Provider-Specific Details:**
- Uses Google's safety settings configuration
- Supports grounding via Google Search
- Custom handling for PDF documents based on size

### GrokAdapter

Adapter for Grok AI (X.AI) models.

**Key Features:**
- Text generation
- Function calling implementation

**Provider-Specific Details:**
- Uses OpenAI-compatible API but with X.AI base URL
- Temperature control but no max_tokens support

### DeepSeekAdapter

Adapter for DeepSeek models.

**Key Features:**
- Basic text generation
- Content type handling (text, images, documents)

**Provider-Specific Details:**
- Special handling for deepseek-reasoner model which doesn't support temperature
- Limited implementation (function calling not fully implemented)

### OpenRouterAdapter

Adapter for accessing multiple LLMs through OpenRouter's API.

**Key Features:**
- Basic text generation
- Content type handling similar to OpenAI

**Provider-Specific Details:**
- Minimal implementation with limited function calling support
- Uses OpenAI-compatible interface but with OpenRouter base URL

### SpeechmaticsAdapter

Adapter for Speechmatics speech recognition service.

**Key Features:**
- Speech-to-text transcription

**Provider-Specific Details:**
- Supports entity detection
- Speaker diarization

### ElevenLabsAdapter

Adapter for ElevenLabs voice synthesis.

**Key Features:**
- Voice selection
- Text-to-speech generation

**Provider-Specific Details:**
- Not a complete LLM adapter, focused only on voice synthesis

## Usage Example

```python
from llm_platform.adapters.anthropic_adapter import AnthropicAdapter
from llm_platform.services.conversation import Conversation

# Initialize the adapter
adapter = AnthropicAdapter()

# Create a conversation
conversation = Conversation(system_prompt="You are a helpful assistant.")
conversation.messages.append({"role": "user", "content": "What is the capital of France?"})

# Make a request
response = adapter.request_llm(
    model="claude-3-opus-20240229",
    the_conversation=conversation,
    temperature=0.7
)

print(response)
```

## Implementing a New Adapter

To implement a new adapter for an LLM provider:

1. Create a new file named `provider_adapter.py` in the adapters directory
2. Inherit from `AdapterBase`
3. Implement at minimum:
   - `convert_conversation_history_to_adapter_format()`
   - `request_llm()`
4. Optionally implement:
   - `request_llm_with_functions()`
   - `generate_image()`
   - Async versions of these methods

The adapter should handle the transformation of the LLM Platform's standard conversation format into the provider's expected format, and vice versa for responses.