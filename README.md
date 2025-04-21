# Ollama Python Client

This is a lightweight, Python client for interacting with a locally hosted [Ollama](https://ollama.com/) API. It allows you to easily send prompts to any installed model and customize generation behavior using supported inference options.

## Features

- Clean and minimal API
- Full control over all supported generation parameters
- Automatically stores the last API response
- Designed for local use via HTTP with the Ollama `/api/generate` endpoint

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ollama-python-client.git
cd ollama-python-client
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

This project depends on:
- `requests`

## Usage

```python
from Ollama import Ollama, Options

ollama = Ollama(base_url="http://10.0.0.1:11434", model="llama3:latest")

# Optional parameters
options = Options(
    temperature=0.7,
    top_p=0.9,
    stop=["\n"]
)

response = ollama.send_prompt("Explain quantum entanglement.", options)
print(response)
```

## Options

The `Options` class supports the following parameters:

- `temperature`
- `top_p`
- `top_k`
- `num_predict`
- `stop`
- `repeat_last_n`
- `repeat_penalty`
- `seed`
- `presence_penalty`
- `frequency_penalty`
- `mirostat`
- `mirostat_tau`
- `mirostat_eta`
- `num_ctx`