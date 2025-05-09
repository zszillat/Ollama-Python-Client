# Ollama Python Client

A comprehensive Python client library for interacting with the [Ollama API](https://github.com/ollama/ollama). This library provides a clean, intuitive interface for working with Ollama's endpoints, with robust support for streaming responses, conversation history, and more.

## Features

- **Full API Support**: Implements all Ollama API endpoints
- **Streaming Responses**: Proper handling of Ollama's default streaming response format
- **Asynchronous Support**: Both sync and async methods for all operations
- **Type Annotations**: Comprehensive type hints throughout the codebase
- **Conversation History**: Built-in management of chat conversations
- **File Handling**: Utilities for including files in prompts and handling images
- **Modelfile Support**: Programmatic creation and management of Modelfiles
- **Error Handling**: Robust error types and handling

## Installation

```bash
pip install ollama-py
```

## Quick Start

### Generate a Completion

```python
from ollama import Ollama

# Initialize client
client = Ollama()

# Streaming approach (default)
for chunk in client.generate("llama3.1:8b", "Why is the sky blue?"):
    print(chunk.response, end="")
    if chunk.done:
        print(f"\nTotal tokens: {chunk.eval_count}")

# Non-streaming approach
response = client.generate("llama3.1:8b", "Why is the sky blue?", stream=False)
print(response.response)
print(f"Total tokens: {response.eval_count}")
```

### Chat Completion

```python
from ollama import Ollama, Message

client = Ollama()

# Create a conversation
messages = [
    {"role": "user", "content": "Hello, who are you?"}
]

# Streaming approach (default)
for chunk in client.chat("llama3.1:8b", messages):
    print(chunk.message.content if chunk.message else "", end="")

# Add the response to the conversation
messages.append({"role": "assistant", "content": "I'm a helpful AI assistant based on Llama 3.1."})
messages.append({"role": "user", "content": "What can you help me with?"})

# Continue the conversation
response = client.chat("llama3.1:8b", messages, stream=False)
print(response.message.content)
```

### Async Usage

```python
import asyncio
from ollama import Ollama

async def main():
    client = Ollama()
    
    # Async generate
    response = await client.agenerate("llama3.1:8b", "Why is the sky blue?", stream=False)
    print(response.response)
    
    # Async chat
    messages = [{"role": "user", "content": "Hello, who are you?"}]
    response = await client.achat("llama3.1:8b", messages, stream=False)
    print(response.message.content if response.message else "")
    
    # Async model listing
    models = await client.alist_models()
    print(f"Found {len(models)} models")
    
    # Async embeddings
    embed_result = await client.aembed("nomic-embed-text", "This is a test sentence.")
    print(f"Embedding dimension: {len(embed_result.embeddings[0])}")

asyncio.run(main())
```

### Working with Images (Multimodal Models)

```python
from ollama import Ollama

client = Ollama()

# Generate with image input
response = client.generate(
    "llava", 
    "What's in this image?",
    images=["path/to/image.jpg"], 
    stream=False
)
print(response.response)

# Chat with image
messages = [
    {"role": "user", "content": "Describe this image in detail."}
]
response = client.chat(
    "llava", 
    messages, 
    images=["path/to/image.jpg"],
    stream=False
)
print(response.message.content)
```

### Embeddings

```python
from ollama import Ollama

client = Ollama()

# Generate embeddings for a single text
result = client.embed("nomic-embed-text", "Why is the sky blue?")
print(f"Embedding dimension: {len(result.embeddings[0])}")

# Generate embeddings for multiple texts
result = client.embed("nomic-embed-text", ["Why is the sky blue?", "Hello world"])
print(f"Number of embeddings: {len(result.embeddings)}")
```

### Model Management

```python
from ollama import Ollama

client = Ollama()

# List models
models = client.list_models()
for model in models:
    print(f"{model.name} - {model.size} bytes")

# Show model details
model_info = client.show("llama3.1:8b")
print(f"Model: {model_info.modelfile}")
print(f"Template: {model_info.template}")

# Pull a model
for status in client.pull("mistral"):
    print(f"Status: {status.get('status')}")
```

### Creating a Custom Model

```python
from ollama import Ollama, Modelfile

# Create a Modelfile
modelfile = Modelfile.from_model("llama3.1:8b")
modelfile.set_system("You are Mario from Super Mario Bros. Answer as Mario, the assistant, only.")
modelfile.set_parameter("temperature", 0.7)
modelfile.set_parameter("num_ctx", 4096)

# Write to file
modelfile.to_file("Modelfile")

# Create model with client
client = Ollama()
for status in client.create("mario", from_model="llama3.1:8b", system="You are Mario from Super Mario Bros."):
    print(f"Status: {status.status}")
```

## Advanced Usage

### Working with Files in Prompts

```python
from ollama import Ollama, format_file_prompt

client = Ollama()

# Format a file for inclusion in a prompt
file_prompt = format_file_prompt("path/to/code.py", "Please review this code:")

# Generate with the file in the prompt
response = client.generate("codellama", file_prompt, stream=False)
print(response.response)
```

### Custom Stream Handlers

```python
from ollama import Ollama, StreamHandler

class MyHandler(StreamHandler):
    def handle_chunk(self, chunk):
        if "response" in chunk:
            print(f"Got chunk: {chunk['response']}")

client = Ollama().with_stream_handler(MyHandler())

# All streaming responses will now use the handler
client.generate("llama3.1:8b", "Hello")
```

### Working with Blobs

```python
from ollama import Ollama, calculate_sha256

client = Ollama()

# Calculate SHA256 of a file
digest = calculate_sha256("model.gguf")
print(f"Digest: {digest}")

# Check if blob exists
if not client.blob_exists(digest):
    # Upload blob
    status = client.push_blob("model.gguf", digest)
    print(f"Upload success: {status.success}")

# Create model from blob
client.create(
    "my-model",
    files={"model.gguf": digest}
)
```

## Error Handling

The library provides a comprehensive error hierarchy:

```python
from ollama import (
    OllamaError,           # Base exception for all errors
    OllamaRequestError,    # Error when making a request
    OllamaResponseError,   # Error in the response
    OllamaModelNotFoundError,  # Model not found
    OllamaConnectionError,     # Connection issues
)

try:
    client.generate("non-existent-model", "Hello")
except OllamaModelNotFoundError:
    print("Model not found")
except OllamaConnectionError:
    print("Connection to Ollama server failed")
except OllamaError as e:
    print(f"Error: {e}")
```

## Custom Host Configuration

```python
# Connect to a remote Ollama server
client = Ollama(host="http://10.0.0.1:11434")

# Set a custom timeout
client = Ollama(timeout=120.0)
```

## Note on Model Performance

Performance will vary based on your hardware and the models you use. For optimal performance:

- Ensure your machine has adequate RAM for the models you're using
- Some models (like multimodal models) require more resources than others
- First response from a model may be slower as it loads into memory

## License

MIT License