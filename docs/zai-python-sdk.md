> ## Documentation Index
> Fetch the complete documentation index at: https://docs.z.ai/llms.txt
> Use this file to discover all available pages before exploring further.

# Official Python SDK

Z.AI Python SDK is the official Python development toolkit provided by Z.AI, offering Python developers convenient and efficient AI model integration solutions.

### Core Advantages

<CardGroup cols={2}>
  <Card title="Simple and Easy" icon="rocket">
    Pythonic API design, comprehensive documentation and examples for quick start
  </Card>

  <Card title="Complete Features" icon="puzzle-piece">
    Supports Z.AI's full range of models, including language, vision, image generation, etc.
  </Card>

  <Card title="High Performance" icon="gauge-high">
    Async support, connection pool management, optimized network request handling
  </Card>

  <Card title="Type Safety" icon="shield-check">
    Complete type hints, IDE-friendly, reducing development errors
  </Card>
</CardGroup>

### Supported Features

* **💬 Chat Conversations**: Support for single-turn and multi-turn conversations, streaming and non-streaming responses
* **🔧 Function Calling**: Enable AI models to call your custom functions
* **👁️ Vision Understanding**: Image analysis, visual understanding
* **🎨 Image Generation**: Generate high-quality images from text descriptions
* **🎬 Video Generation**: Creative content generation from text to video
* **🔊 Speech Processing**: Speech-to-text, text-to-speech
* **📊 Text Embedding**: Text vectorization, supporting semantic search
* **🤖 Intelligent Assistants**: Build professional AI assistant applications
* **🛡️ Content Moderation**: Text and image content safety detection

## Technical Specifications

### Environment Requirements

* **Python Version**: Python 3.8 or higher
* **Package Manager**: pip or poetry
* **Network Requirements**: HTTPS connection support
* **API Key**: Valid Z.AI API key required

### Dependency Management

The SDK adopts a modular design, allowing you to selectively install functional modules as needed:

* **Core Module**: Basic API calling functionality
* **Async Module**: Asynchronous and concurrent processing support
* **Utility Module**: Utility tools and auxiliary functions

## Quick Start

### Environment Requirements

<CardGroup cols={2}>
  <Card title="Python Version" icon="python">
    Python 3.8 or higher
  </Card>

  <Card title="Package Manager" icon="building">
    poetry (recommended), uv (recommended), pip
  </Card>
</CardGroup>

<Warning>
  Supports Python 3.8, 3.9, 3.10, 3.11, 3.12 versions, cross-platform compatible with Windows, macOS, Linux
</Warning>

### Install SDK

#### Install using pip

```bash theme={null}
# Install latest version
pip install zai-sdk

# Or specify version
pip install zai-sdk==0.2.3
```

#### Verify Installation

```python theme={null}
import zai
print(zai.__version__)
```

### Get API Key

1. Access [Z.AI Open Platform](https://z.ai/model-api), Register or Login.
2. Create an API Key in the [API Keys](https://z.ai/manage-apikey/apikey-list) management page.
3. Copy your API Key for use.

<Tip>
  It is recommended to set the API Key as an environment variable: `export ZAI_API_KEY=your-api-key`
</Tip>

<Tip>
  Domestic Z.AI platform uses ZaiClient
</Tip>

```
Domestic API URL: https://api.z.ai/api/paas/v4/
```

#### Create Client

<Tabs>
  <Tab title="Environment Variable">
    ```python theme={null}
    from zai import ZaiClient
    import os

    # Read API Key from environment variable
    client = ZaiClient(api_key=os.getenv("ZAI_API_KEY"))

    # Or use directly (if environment variable is set)
    client = ZaiClient()

    ```
  </Tab>

  <Tab title="Direct Setting">
    ```python theme={null}
    from zai import ZaiClient, ZaiClient

    # Set API Key directly
    client = ZaiClient(api_key="abc123.def456")

    ```
  </Tab>
</Tabs>

#### Basic Conversation

```python theme={null}
from zai import ZaiClient

# Initialize client
client = ZaiClient(api_key="your-api-key")

# Create chat completion
response = client.chat.completions.create(
    model="glm-5.2",
    messages=[
        {"role": "user", "content": "Hello, please introduce yourself, Z.ai!"}
    ]
)
print(response.choices[0].message.content)
```

#### Streaming Conversation

```python theme={null}
# Create streaming chat request
from zai import ZaiClient

# Initialize client
client = ZaiClient(api_key="your-api-key")

# Create chat completion
response = client.chat.completions.create(
    model='glm-5.2',
    messages=[
        {'role': 'system', 'content': 'You are an AI writer.'},
        {'role': 'user', 'content': 'Tell a story about AI.'},
    ],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='')
```

#### Multi-turn Conversation

```python theme={null}
from zai import ZaiClient
client = ZaiClient(api_key="your-api-key")
response = client.chat.completions.create(
    model="glm-5.2",  # Please fill in the model name you want to call
    messages=[
        {"role": "user", "content": "As a marketing expert, please create an attractive slogan for my product"},
        {"role": "assistant", "content": "Of course, to create an attractive slogan, please tell me some information about your product"},
        {"role": "user", "content": "Z.AI Open Platform"},
        {"role": "assistant", "content": "Ignite the future, Z.AI draws infinite possibilities, making innovation within reach!"},
        {"role": "user", "content": "Create a more precise and attractive slogan"}
    ],
)
print(response.choices[0].message.content)
```

### Complete Example

```python theme={null}
from zai import ZaiClient
import os

def main():
    # Initialize client
    client = ZaiClient(api_key=os.getenv("ZAI_API_KEY"))
    
    print("Welcome to Z.ai Chatbot! Type 'quit' to exit.")
    
    # Conversation history
    conversation = [
        {"role": "system", "content": "You are a friendly AI assistant"}
    ]
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            break
        
        try:
            # Add user message
            conversation.append({"role": "user", "content": user_input})
            
            # Create chat request
            response = client.chat.completions.create(
                model="glm-5.2",
                messages=conversation,
                temperature=1.0,
                max_tokens=1000
            )
            
            # Get AI response
            ai_response = response.choices[0].message.content
            print(f"AI: {ai_response}")
            
            # Add AI response to conversation history
            conversation.append({"role": "assistant", "content": ai_response})
            
        except Exception as e:
            print(f"Error occurred: {e}")
    
    print("Goodbye!")

if __name__ == "__main__":
    main()
```

### Error Handling

```python theme={null}
from zai import ZaiClient
import zai

def robust_chat(message):
    client = ZaiClient(api_key="your-api-key")
    
    try:
        response = client.chat.completions.create(
            model="glm-5.2",
            messages=[{"role": "user", "content": message}]
        )
        return response.choices[0].message.content
        
    except zai.core.APIStatusError as err:
        return f"API status error: {err}"
    except zai.core.APITimeoutError as err:
        return f"Request timeout: {err}"
    except Exception as err:
        return f"Other error: {err}"

# Usage example
result = robust_chat("Hello")
print(result)
```

### Advanced Configuration

```python theme={null}
import httpx
from zai import ZaiClient

# Custom HTTP client
httpx_client = httpx.Client(
    limits=httpx.Limits(
        max_keepalive_connections=20,
        max_connections=100
    ),
    timeout=30.0
)

# Create client with custom configuration
client = ZaiClient(
    api_key="your-api-key",
    base_url="https://api.z.ai/api/paas/v4/",
    timeout=httpx.Timeout(timeout=300.0, connect=8.0),
    max_retries=3,
    http_client=httpx_client
)
```

## Advanced Features

### Function Calling

Function calling allows AI models to call functions you define to get real-time information or perform specific operations.

#### Defining and Using Functions

```python theme={null}
from zai import ZaiClient
import json

# Define functions
def get_weather(location, date=None):
    """Get weather information"""
    # Simulate weather API call
    return {
        "location": location,
        "date": date or "today",
        "weather": "sunny",
        "temperature": "25°C",
        "humidity": "60%"
    }

def get_stock_price(symbol):
    """Get stock price"""
    # Simulate stock API call
    return {
        "symbol": symbol,
        "price": 150.25,
        "change": "+2.5%"
    }

# Function descriptions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information for a specified location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location name"
                    },
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get current stock price",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol"
                    }
                },
                "required": ["symbol"]
            }
        }
    }
]

# Use function calling
client = ZaiClient(api_key="your-api-key")

response = client.chat.completions.create(
    model='glm-5.2',
    messages=[
        {'role': 'user', 'content': 'How\'s the weather in Beijing today?'}
    ],
    tools=tools,
    tool_choice="auto"
)

# Handle function calling
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        if function_name == "get_weather":
            result = get_weather(**function_args)
            print(f"Weather info: {result}")
        elif function_name == "get_stock_price":
            result = get_stock_price(**function_args)
            print(f"Stock info: {result}")
else:
    print(response.choices[0].message.content)
```

### Web Search Tool

```python theme={null}
from zai import ZaiClient

# Initialize client
client = ZaiClient(api_key="your-api-key")

# Use web search tool
response = client.chat.completions.create(
    model='glm-5.2',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'What is artificial intelligence?'},
    ],
    tools=[
        {
            'type': 'web_search',
            'web_search': {
                'search_query': 'What is artificial intelligence?',
                'search_result': True,
            },
        }
    ],
    temperature=0.5,
    max_tokens=2000,
)

print(response)
```

### Video Generation

```python theme={null}
from zai import ZaiClient
import time

client = ZaiClient(api_key="your-api-key")

# Submit generation task
response = client.videos.generations(
    model="cogvideox-3",  # Video generation model to use
    image_url=image_url,  # Provided image URL or Base64 encoding
    prompt="Make the scene come alive",
    quality="speed",  # Output mode: "quality" for quality priority, "speed" for speed priority
    with_audio=True,
    size="1920x1080",  # Video resolution, supports up to 4K (e.g., "3840x2160")
    fps=30,  # Frame rate, can be 30 or 60
)
print(response)

# Get generation result
time.sleep(60)  # Wait for a while to ensure video generation is complete
result = client.videos.retrieve_videos_result(id=response.id)
print(result)
```

### Streaming Processing

```python theme={null}
class StreamProcessor:
    def __init__(self, client):
        self.client = client
        self.full_response = ""
    
    def stream_chat(self, messages, model="glm-5.2", callback=None):
        """Streaming chat processing"""
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
        
        self.full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                self.full_response += content
                
                if callback:
                    callback(content, self.full_response)
                else:
                    print(content, end="", flush=True)
        
        print()  # New line
        return self.full_response

# Usage example
processor = StreamProcessor(client)

# Custom callback function
def on_token_received(token, full_text):
    # You can implement real-time processing logic here
    print(token, end="", flush=True)

response = processor.stream_chat(
    messages=[{"role": "user", "content": "Write a Python function to calculate Fibonacci sequence"}],
    callback=on_token_received
)
```

## Getting Help

<CardGroup cols={2}>
  <Card title="GitHub Repository" icon="github" href="https://github.com/zai-org/z-ai-sdk-python">
    View source code, submit issues, contribute
  </Card>

  <Card title="API Reference" icon="book" href="/api-reference">
    View complete API documentation
  </Card>

  <Card title="Example Projects" icon="code" href="https://github.com/zai-org/z-ai-sdk-python/tree/main/examples">
    Browse more practical application examples
  </Card>

  <Card title="Best Practices" icon="star" href="https://github.com/zai-org/z-ai-sdk-python">
    Learn best practices for SDK usage
  </Card>
</CardGroup>

<Note>
  This SDK is developed based on the latest API specifications from Z.AI, ensuring synchronization with platform features. It is recommended to regularly update to the latest version for the best experience.
</Note>
