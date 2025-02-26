# Conversation Module Documentation

## Overview

The `conversation.py` module provides classes to manage conversations with language models in the LLM Platform. It defines the data structures for representing messages, function calls, function responses, and thinking responses in a unified format that can be converted to provider-specific formats.

## Classes

### Conversation

The `Conversation` class represents a complete conversation history that can be sent to language models.

#### Constructor

```python
def __init__(self, messages: List[Message] = [], system_prompt: str = None)
```

**Parameters:**
- `messages`: List of Message objects representing the conversation history
- `system_prompt`: System prompt to set context for the language model

#### Properties

- `usage_total`: Returns a dictionary with the total token usage across all messages
- `usage_last`: Returns a dictionary with the token usage of the last message

#### Methods

```python
def clear(self)
```
Clears all messages from the conversation.

```python
def __str__(self)
```
Returns a string representation of the conversation.

### Message

The `Message` class represents a single message in the conversation, which can be from a user, assistant, or function.

#### Constructor

```python
def __init__(self, role: str, content: str, thinking_responses: List[ThinkingResponse]=[], 
             usage: Dict=None, files: List[BaseFile]=[], function_calls: List[FunctionCall]=[],
             function_responses: List[FunctionResponse]=[])
```

**Parameters:**
- `role`: The role of the message sender ("user", "assistant", or "function")
- `content`: The text content of the message
- `thinking_responses`: List of ThinkingResponse objects (for models like Claude that support thinking steps)
- `usage`: Dictionary containing token usage information
- `files`: List of BaseFile objects attached to the message
- `function_calls`: List of FunctionCall objects representing function/tool calls made by the assistant
- `function_responses`: List of FunctionResponse objects representing responses to function/tool calls

### FunctionCall

The `FunctionCall` class represents a function or tool call made by the language model.

#### Constructor

```python
def __init__(self, id: str, name: str, arguments: Dict | List[Dict])
```

**Parameters:**
- `id`: Unique identifier for the function call
- `name`: Name of the function being called
- `arguments`: Arguments passed to the function

#### Methods

```python
@classmethod
def from_openai(cls, tool_call)
```
Creates a FunctionCall instance from an OpenAI tool call.

```python
def to_openai(self) -> Dict
```
Converts the FunctionCall to OpenAI's format.

```python
def to_anthropic(self) -> Dict
```
Converts the FunctionCall to Anthropic's format.

### FunctionResponse

The `FunctionResponse` class represents a response from a function or tool call.

#### Constructor

```python
def __init__(self, name: str, response: Dict, id: str=None)
```

**Parameters:**
- `name`: Name of the function that was called
- `response`: The response data from the function execution
- `id`: Identifier of the function call this response is for

#### Methods

```python
def to_openai(self) -> Dict
```
Converts the FunctionResponse to OpenAI's format.

```python
def to_anthropic(self) -> Dict
```
Converts the FunctionResponse to Anthropic's format.

### ThinkingResponse

The `ThinkingResponse` class represents a thinking/reasoning step from models that support it (like Claude).

#### Constructor

```python
def __init__(self, content: str, id: str=None)
```

**Parameters:**
- `content`: The thinking content
- `id`: Optional identifier for the thinking response

#### Methods

```python
def to_anthropic(self) -> Dict
```
Converts the ThinkingResponse to Anthropic's format.

## Usage Examples

### Creating a Simple Conversation

```python
from llm_platform.services.conversation import Conversation, Message

# Create a new conversation with a system prompt
conversation = Conversation(system_prompt="You are a helpful assistant.")

# Add a user message
user_message = Message(role="user", content="What's the weather like today?")
conversation.messages.append(user_message)

# Add an assistant message
assistant_message = Message(role="assistant", content="I don't have access to real-time weather information. Please check a weather service for accurate information.")
conversation.messages.append(assistant_message)

# Print the conversation
print(conversation)
```

### Working with Function Calls

```python
from llm_platform.services.conversation import Conversation, Message, FunctionCall, FunctionResponse

# Create a conversation
conversation = Conversation(system_prompt="You are a helpful assistant that can use tools.")

# Add a user message
user_message = Message(role="user", content="What's 25 × 4?")
conversation.messages.append(user_message)

# Create a function call (as if made by the assistant)
function_call = FunctionCall(
    id="call_1",
    name="calculator",
    arguments='{"expression": "25 * 4"}'
)

# Create an assistant message with the function call
assistant_message = Message(
    role="assistant",
    content="I'll calculate that for you.",
    function_calls=[function_call]
)
conversation.messages.append(assistant_message)

# Create a function response
function_response = FunctionResponse(
    name="calculator",
    response={"result": 100},
    id="call_1"
)

# Add the function response to the conversation
function_message = Message(
    role="function",
    content="",
    function_responses=[function_response]
)
conversation.messages.append(function_message)

# Add the final assistant message
final_message = Message(
    role="assistant",
    content="The result of 25 × 4 is 100."
)
conversation.messages.append(final_message)
```

### Adding Files to Messages

```python
from llm_platform.services.conversation import Conversation, Message
from llm_platform.services.files import ImageFile

# Create a conversation
conversation = Conversation(system_prompt="You are a helpful assistant.")

# Create an image file
image = ImageFile(file_path="example.jpg")

# Add a user message with the image
user_message = Message(
    role="user",
    content="What's in this image?",
    files=[image]
)
conversation.messages.append(user_message)
```

### Using Thinking Responses (Claude-specific)

```python
from llm_platform.services.conversation import Conversation, Message, ThinkingResponse

# Create a conversation
conversation = Conversation(system_prompt="You are a helpful assistant.")

# Add a user message
user_message = Message(role="user", content="What's the square root of 144?")
conversation.messages.append(user_message)

# Create a thinking response
thinking = ThinkingResponse(
    content="To find the square root of 144, I need to find a number that when multiplied by itself equals 144. 12 × 12 = 144, so the square root of 144 is 12.",
    id="thinking_1"
)

# Add an assistant message with thinking
assistant_message = Message(
    role="assistant",
    content="The square root of 144 is 12.",
    thinking_responses=[thinking]
)
conversation.messages.append(assistant_message)
```

## Format Conversion

The classes in this module provide methods to convert to different provider-specific formats:

- `to_openai()`: Converts to OpenAI's API format
- `to_anthropic()`: Converts to Anthropic's API format

This allows the LLM Platform to maintain a unified conversation format internally while communicating with different model providers using their expected formats.