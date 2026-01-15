# APIHandler Documentation

## Overview

The `APIHandler` class is the central component of the LLM Platform that manages interactions with various language model providers. It provides a unified interface for making synchronous and asynchronous requests to language models, converting speech to text, generating images, and retrieving available models.

## Class: APIHandler

### Constructor

```python
def __init__(self, system_prompt: str = "You are a helpful assistant")
```

**Parameters:**
- `system_prompt (str)`: Initial system prompt for the conversation (default: "You are a helpful assistant")

### Key Attributes

- `adapters (Dict[str, Any])`: Dictionary storing initialized model adapters
- `model_config (ModelConfig)`: Configuration for the language models
- `the_conversation (Conversation)`: The conversation context

### Methods

#### Request Methods

```python
def request(self, model: str, prompt: str, functions: Union[List[BaseTool], List[Callable]] = None,
           files: List[BaseFile] = [], tool_output_callback: Callable = None,
           additional_parameters: AdditionalParameters | None = None) -> str
```

Makes a synchronous request to the language model.

**Parameters:**
- `model (str)`: The name of the model to use
- `prompt (str)`: The prompt to send to the model
- `functions (Union[List[BaseTool], List[Callable]])`: Optional tools or callables to use
- `files (List[BaseFile])`: Optional files to include in the request
- `tool_output_callback (Callable)`: Optional callback function for tool output
- `additional_parameters (AdditionalParameters)`: Optional additional parameters (e.g. `temperature`, `max_tokens`, `reasoning`, `text.verbosity`, `web_search`)

**Returns:**
- `str`: The response from the language model

```python
async def request_async(self, model: str, prompt: str, functions: Union[List[BaseTool], List[Callable]] = None,
                       files: List[BaseFile] = [], tool_output_callback: Callable = None,
                       additional_parameters: AdditionalParameters | None = None) -> str
```

Makes an asynchronous request to the language model with the same parameters as `request()`. Provider tuning flows through `additional_parameters`.

#### Adapter Management

```python
def _lazy_initialization_of_adapter(self, adapter_name: str)
```

Lazily initializes and returns the specified adapter.

**Parameters:**
- `adapter_name (str)`: The name of the adapter to initialize

**Returns:**
- The initialized adapter

**Raises:**
- `ValueError`: If the specified adapter is not supported

```python
def get_adapter(self, model_name: str) -> Any
```

Gets the appropriate adapter for the given model name.

**Parameters:**
- `model_name (str)`: The name of the model

**Returns:**
- The adapter for the specified model

#### Utility Methods

```python
@staticmethod
def calculate_tokens(text) -> Dict[str, int]
```

Calculates the number of tokens in the given text.

**Parameters:**
- `text (str)`: The text to calculate tokens for

**Returns:**
- `Dict[str, int]`: A dictionary containing the byte count and token count

#### Speech and Image Generation

```python
def voice_to_text(self, audio_file: BinaryIO, audio_format: str, provider: str = 'openai', **kwargs)
```

Converts speech to text using the specified provider.

**Parameters:**
- `audio_file (BinaryIO)`: The audio file to convert
- `audio_format (str)`: The format of the audio file
- `provider (str)`: The provider to use (default: 'openai', also supports 'speechmatics')
- `**kwargs`: Additional provider-specific parameters

**Returns:**
- `str`: The transcribed text

```python
def voice_file_to_text(self, audio_file_name: str, provider: str = 'openai', **kwargs)
```

Converts a voice file to text.

**Parameters:**
- `audio_file_name (str)`: The path to the audio file
- `provider (str)`: The provider to use (default: 'openai', also supports 'speechmatics')
- `**kwargs`: Additional provider-specific parameters

**Returns:**
- `str`: The transcribed text

```python
def generate_image(self, prompt: str, provider: str = 'openai', n = 1, **kwargs)
```

Generates an image based on the given prompt.

**Parameters:**
- `prompt (str)`: A textual description of the desired image
- `provider (str)`: The provider to use (default: 'openai', also supports 'google')
- `n (int)`: The number of images to generate (default: 1)
- `**kwargs`: Additional provider-specific parameters

**Returns:**
- `Union[str, List[str]]`: The generated image URL(s) or list of images

#### Model Management

```python
def get_models(self, adapter_name: str) -> List[str]
```

Retrieves available models for the specified adapter.

**Parameters:**
- `adapter_name (str)`: The name of the adapter

**Returns:**
- `List[str]`: List of available model names

## Supported Adapters

The `APIHandler` supports the following adapters:
- OpenAIAdapter
- AnthropicAdapter
- OpenRouterAdapter
- SpeechmaticsAdapter
- GoogleAdapter
- GrokAdapter
- DeepSeekAdapter

## Usage Examples

### Basic Text Request

```python
from llm_platform.core.llm_handler import APIHandler

# Initialize the handler
handler = APIHandler(system_prompt="You are a helpful AI assistant.")

# Make a simple request
response = handler.request(
    model="gpt-4",
    prompt="What is the capital of France?"
)
print(response)
```

### Using Tools/Functions

```python
from llm_platform.core.llm_handler import APIHandler
from llm_platform.tools.base import BaseTool

# Define a custom tool
class CalculatorTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Performs basic arithmetic operations"
        )
    
    def execute(self, expression: str) -> str:
        return str(eval(expression))

# Initialize the handler
handler = APIHandler()

# Make a request with the tool
response = handler.request(
    model="claude-3-sonnet",
    prompt="Calculate 25 * 4",
    functions=[CalculatorTool()]
)
print(response)
```

### File-based Request

```python
from llm_platform.core.llm_handler import APIHandler
from llm_platform.services.files import PDFDocumentFile

# Initialize the handler
handler = APIHandler()

# Create a file object
pdf_file = PDFDocumentFile(file_path="document.pdf")

# Make a request with the file
response = handler.request(
    model="gpt-4-vision",
    prompt="Summarize this document",
    files=[pdf_file]
)
print(response)
```

### Speech to Text

```python
from llm_platform.core.llm_handler import APIHandler

# Initialize the handler
handler = APIHandler()

# Convert speech file to text
transcript = handler.voice_file_to_text(
    audio_file_name="recording.mp3",
    provider="openai"
)
print(transcript)
```

### Image Generation

```python
from llm_platform.core.llm_handler import APIHandler

# Initialize the handler
handler = APIHandler()

# Generate an image
image_url = handler.generate_image(
    prompt="A serene mountain landscape at sunset",
    provider="openai",
    size="1024x1024"
)
print(image_url)
```
