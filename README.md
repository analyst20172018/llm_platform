# LLM Platform

A unified Python library for interacting with various LLM providers through a consistent interface.

## Overview

LLM Platform is a Python library that provides a unified API for interacting with multiple LLM providers, including OpenAI, Anthropic, Google, DeepSeek, Grok, OpenRouter, and more. The platform abstracts away the differences between LLM providers, allowing developers to focus on application logic while maintaining the flexibility to switch between models as needed.

## Features

- **Unified API Interface**: Interact with any supported LLM through the same API
- **Multi-modal Support**: Process images, PDFs, Excel files, and text documents
- **Function/Tool Calling**: Enable LLMs to execute code and tools
- **Voice-to-Text Conversion**: Transcribe audio using OpenAI or Speechmatics
- **Image Generation**: Create images from text using OpenAI or Google
- **Conversation Management**: Track conversation history and token usage
- **Asynchronous Support**: Make asynchronous requests to LLM providers

## Installation

```bash
# Installation instructions to be added
```

## Environment Setup

1. Create a `.env` file in your project root with API keys for the providers you plan to use:

```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
GROK_API_KEY=your_grok_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
SPEECHMATICS_API_KEY=your_speechmatics_api_key
```

## Basic Usage

### Initializing the API Handler

```python
from llm_platform.core.llm_handler import APIHandler

# Initialize with a system prompt
handler = APIHandler(system_prompt="You are a helpful assistant.")
```

### Making a Simple Text Request

```python
# Make a request to a specific model
response = handler.request(
    model="claude-3-5-sonnet-latest",
    prompt="What is the capital of France?",
    temperature=0
)

print(response)
```

### Using Different Models

```python
# OpenAI model
openai_response = handler.request(
    model="gpt-4o", 
    prompt="Explain quantum computing in simple terms",
    temperature=0
)

# Google model
google_response = handler.request(
    model="gemini-2.0-pro", 
    prompt="What are the main challenges in AI safety?",
    temperature=0
)

# Anthropic model
anthropic_response = handler.request(
    model="claude-3-opus-latest", 
    prompt="Write a short story about a robot learning to paint",
    temperature=0.7
)
```

### Working with Images

```python
from llm_platform.services.conversation import ImageFile

# Request with image
response = handler.request(
    model="gpt-4o",
    prompt="What is in this image?",
    files=[ImageFile.from_path("/path/to/image.jpg")]
)

print(response)
```

### Generating Images

```python
# Generate an image with OpenAI
openai_image = handler.generate_image(
    prompt="A cyberpunk cityscape with neon lights",
    provider="openai",
    n=1,
    size="1024x1024",
    quality="standard"
)

# Generate an image with Google
google_image = handler.generate_image(
    prompt="A serene mountain landscape at sunset",
    provider="google",
    n=1,
    aspect_ratio="16:9"
)
```

### Function/Tool Calling

```python
from llm_platform.tools.powershell import RunPowerShellCommand

# Define a callback function to handle tool outputs
def tool_callback(tool_name, tool_args, tool_response):
    print(f"Tool {tool_name} called with {tool_args}")
    print(f"Tool response: {tool_response}")

# Make a request with function calling
response = handler.request(
    model="gpt-4o", 
    prompt="Get the specs of my computer",
    functions=[RunPowerShellCommand()],
    temperature=0,
    tool_output_callback=tool_callback
)

print(response)
```

### Voice to Text Conversion

```python
# Convert audio file to text using OpenAI
transcript = handler.voice_file_to_text(
    audio_file_name="/path/to/audio.mp3",
    provider="openai",
    language="en"
)

print(transcript)

# Convert audio file to text using Speechmatics
transcript = handler.voice_file_to_text(
    audio_file_name="/path/to/audio.mp3",
    provider="speechmatics",
    language="en"
)

print(transcript)
```

## Core Components

### APIHandler

The central class that manages interactions with LLM providers:

```python
class APIHandler:
    def __init__(self, system_prompt="You are a helpful assistant", logging_level=logging.INFO):
        # Initialize the API handler
        
    def request(self, model, prompt, functions=None, files=[], temperature=0, 
                tool_output_callback=None, additional_parameters={}, **kwargs):
        # Make a synchronous request to the LLM
        
    async def request_async(self, model, prompt, functions=None, files=[], 
                           temperature=0, tool_output_callback=None, 
                           additional_parameters={}, **kwargs):
        # Make an asynchronous request to the LLM
        
    def voice_to_text(self, audio_file, audio_format, provider='openai', **kwargs):
        # Convert audio to text
        
    def generate_image(self, prompt, provider='openai', n=1, **kwargs):
        # Generate an image
        
    def get_models(self, adapter_name):
        # Get available models for a specific adapter
```

### Conversation Management

```python
from llm_platform.services.conversation import Conversation, Message

# Create a conversation
conversation = Conversation(system_prompt="You are a helpful assistant")

# Add a user message
message = Message(role="user", content="Hello, how are you?")
conversation.messages.append(message)

# Get conversation usage statistics
total_usage = conversation.usage_total
last_usage = conversation.usage_last
```

### File Handling

```python
from llm_platform.services.files import TextDocumentFile, PDFDocumentFile, ImageFile, AudioFile

# Create file objects
text_file = TextDocumentFile.from_path("/path/to/text.txt")
pdf_file = PDFDocumentFile.from_path("/path/to/document.pdf")
image_file = ImageFile.from_path("/path/to/image.jpg")
audio_file = AudioFile.from_path("/path/to/audio.mp3")

# Make a request with files
response = handler.request(
    model="gpt-4o",
    prompt="Summarize this document and describe the image",
    files=[pdf_file, image_file],
    temperature=0
)
```

### Creating Custom Tools

```python
from llm_platform.tools.base import BaseTool
from pydantic import BaseModel, Field

class MyCustomTool(BaseTool):
    """
    A custom tool that performs a specific function.
    """
    
    class InputModel(BaseModel):
        param1: str = Field(description="First parameter", required=True)
        param2: int = Field(description="Second parameter", required=False)
        
    def __call__(self, param1, param2=0):
        # Implement the tool's functionality
        result = f"Processed {param1} with parameter {param2}"
        return {"result": result}

# Use the custom tool
response = handler.request(
    model="claude-3-5-sonnet-latest",
    prompt="Use my custom tool with 'test' as the first parameter",
    functions=[MyCustomTool()],
    temperature=0
)
```

## Supported Providers

- **OpenAI**: GPT models (GPT-4o, etc.) and "o1" models
- **Anthropic**: Claude models (Claude 3 series)
- **Google**: Gemini models (Gemini 2.0 Pro, Flash, etc.)
- **DeepSeek**: DeepSeek models (DeepSeek Chat, DeepSeek Reasoner)
- **Grok**: Grok models (Grok 2)
- **OpenRouter**: Various models via OpenRouter (e.g., Llama models)
- **Speechmatics**: Speech-to-text services
- **ElevenLabs**: Text-to-speech services

## License

[License information to be added]

## Contributing

[Contribution guidelines to be added]