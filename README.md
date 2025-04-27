# Ollama Client & Web App

A Python client and web interface for working with a locally run Ollama API.

## Features
- Send prompts and manage sessions with Python.
- Web app for chatting, switching models, and saving conversations.
- Full control over generation options.
- Local-first, no external servers.

## Installation

```bash
git clone https://github.com/zszillat/OllamaPythonClient.git
cd OllamaPythonClient
pip install -r requirements.txt
```

## Running the Web App

```bash
cd web_app
uvicorn app:app --reload
```

Go to [http://localhost:8000](http://localhost:8000) in your browser.