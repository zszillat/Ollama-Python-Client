# Ollama Python Client

A simple Python wrapper for interacting with a locally hosted Ollama server. Supports both text and multimodal (image) models, full chat history, customizable generation parameters, and saving/loading conversations.

## Features

- Send chat-style prompts with history preservation
- Configure every Ollama `/api/generate` and `/api/chat` parameter via `Options`
- Multimodal support: attach Base64-encoded images
- Export and reload conversation sessions to/from JSON files
- Clear chat history on demand

## Installation

1. Clone this repository or copy `Ollama.py` into your project.
2. Install dependencies:

   ```bash
   pip install requests
   ```

## Usage

```python
from Ollama import Options, Ollama

# Start a new session
client = Ollama('http://localhost:11434', 'deepseek-coder-v2:16b')

# Create generation options (optional)
opts = Options(temperature=0.7, top_p=0.9, num_predict=256)

# Send a prompt
response = client.send_prompt('Hello, how are you?', options=opts)
print(response)

# Attach an image (Base64 string) to a multimodal model
# image_data = '<base64-encoded-data>'
# response = client.send_prompt('Describe this image', images=[image_data])

# Export session
client.export_conversation('session1.json')

# Later, reload session:
client2 = Ollama('session1.json')
print(client2.send_prompt('Continuing our chat...'))
```

## API Reference

### `Options`
All parameters default to `None` (omitted). Only non-null fields are sent.
- `temperature`, `top_p`, `top_k`, `num_predict`, `stop`, `repeat_penalty`, etc.
- Advanced sampling: `num_keep`, `typical_p`, `min_p`, `penalize_newline`, etc.
- Hardware controls: `num_gpu`, `num_thread`, `low_vram`, `use_mmap`, etc.

### `Ollama` class

- `__init__(base_url, model)` – start new session
- `__init__(file_path)` – load from exported JSON
- `send_prompt(prompt, options=None, suffix=None, system=None, template=None, raw=None, keep_alive=None, images=None, format=None)` – send chat message
- `clear_history()` – reset conversation history
- `export_conversation(file_path)` – save session to JSON