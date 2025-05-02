# === Ollama.py ===
import json
import requests
from typing import Any, Dict, List, Optional, Union
from .Options import Options

class Ollama:
    def __init__(self, base_url_or_file: str, model: Optional[str] = None):
        # Initialize client or load existing session
        if model:
            self.base_url, self.model = base_url_or_file.rstrip("/"), model
            self.history, self.last_request = [], None
        else:
            data = json.load(open(base_url_or_file, encoding="utf-8"))
            self.base_url, self.model = data["base_url"].rstrip("/"), data["model"]
            self.history, self.last_request = data.get("history", []), data.get("last_request")

    def get_installed_models(self) -> List[str]:
        # Fetch available model names
        data = requests.get(f"{self.base_url}/api/models").json()
        return [m["name"] for m in data.get("models", [])]

    def add_message(self, role: str, content: str) -> None:
        # Record a chat message
        self.history.append({"role": role, "content": content})

    def clear_history(self) -> None:
        # Reset conversation history
        self.history.clear()

    def send_prompt(
        self,
        prompt: str,
        options: Optional[Options] = None,
        **kwargs: Any
    ) -> str:
        # Send user prompt and return assistant response
        self.add_message("user", prompt)
        payload: Dict[str, Any] = {"model": self.model, "messages": self.history}
        # Include only provided optional parameters
        payload.update({k: v for k, v in kwargs.items() if v is not None})
        if options:
            opts = options.to_dict()
            if opts:
                payload.update(opts)
        payload["stream"] = False

        resp_json = requests.post(f"{self.base_url}/api/chat", json=payload).json()
        self.last_request = resp_json
        # Extract text content
        msg = resp_json.get("message") or {}
        resp = msg.get("content") or resp_json.get("response", "")
        self.add_message("assistant", resp)
        return resp

    def export_conversation(self, file_path: str) -> None:
        # Save session data to JSON file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({
                "base_url": self.base_url,
                "model": self.model,
                "history": self.history,
                "last_request": self.last_request
            }, f, ensure_ascii=False, indent=2)


# === app.py ===
from fastapi import FastAPI, Request, Form, Query, HTTPException, Body
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import json

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
SETTINGS_PATH = "settings.json"

@app.post("/save_settings")
async def save_settings(new_settings: dict = Body(...)):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(new_settings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise HTTPException(500, f"Failed to save settings: {e}")
    return {"status": "success"}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/delete_chat")
async def delete_chat(filename: str = Form(...)):
    # TODO: implement deletion logic
    return RedirectResponse("/", status_code=303)

@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, user_input: str = Form(...)):
    # Render chat with new input
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/new_chat")
async def new_chat():
    # Start a fresh chat
    return RedirectResponse("/", status_code=303)

@app.get("/load_chat")
async def load_chat(filename: str = Query(...)):
    # TODO: implement load logic
    return RedirectResponse("/", status_code=303)

@app.get("/settings", response_class=HTMLResponse)
async def settings(request: Request):
    # Load settings and presets for the settings page
    settings = json.load(open(SETTINGS_PATH, encoding="utf-8"))
    presets = json.load(open("presets.json", encoding="utf-8"))
    return templates.TemplateResponse(
        "settings.html",
        {"request": request, "settings": settings, "presets": presets}
    )

# === SessionManager.py ===
import json
from OllamaPythonClient.ollama_client import Ollama

class SessionManager:
    def __init__(self):
        # Manages current Ollama client instance
        self.client = None

    @staticmethod
    def load_json(path: str) -> dict:
        # Read JSON from file
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def new_chat(self, url: str = "http://10.0.0.1:11434", model: str = None):
        # Initialize a new chat session
        self.client = Ollama(url, model)

    @staticmethod
    def get_settings() -> dict:
        # Retrieve application settings
        return SessionManager.load_json("settings.json")
