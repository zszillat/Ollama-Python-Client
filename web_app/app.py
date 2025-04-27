from fastapi import FastAPI, Request, Form, Query, Body
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import glob
import json

from ollama_client import Ollama

# Config
BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "tinyllama"

app = FastAPI()

# Paths
current_dir = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))
static_dir = os.path.join(current_dir, "static")
conversations_dir = os.path.join(current_dir, "conversations")
settings_file = os.path.join(current_dir, "settings.json")

app.mount("/static", StaticFiles(directory=static_dir), name="static")
os.makedirs(conversations_dir, exist_ok=True)

# Chat Manager
class ChatManager:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self.reset()

    def reset(self, file_path: str = None):
        if file_path and os.path.exists(file_path):
            self.client = Ollama(file_path)
            self.conversation_file = file_path
        else:
            self.client = Ollama(self.base_url, model=self.model)
            self.conversation_file = os.path.join(conversations_dir, "chat_001.json")

    def new_chat(self):
        conversations = list_conversations()
        new_number = len(conversations) + 1
        filename = f"chat_{new_number:03d}.json"
        self.client = Ollama(self.base_url, model=self.model)
        self.conversation_file = os.path.join(conversations_dir, filename)

chat_manager = ChatManager(BASE_URL, DEFAULT_MODEL)

# Utilities
def list_conversations():
    files = glob.glob(os.path.join(conversations_dir, "*.json"))
    files.sort()
    return [os.path.basename(file) for file in files]

def load_settings():
    with open(settings_file, 'r') as f:
        return json.load(f)

# API Endpoints
@app.post("/save_settings")
async def save_settings(new_settings: dict = Body(...)):
    with open(settings_file, 'w') as f:
        json.dump(new_settings, f, indent=4)
    return {"status": "success"}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    conversations = list_conversations()
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "messages": chat_manager.client.history,
        "conversations": conversations
    })

@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, user_input: str = Form(...)):
    chat_manager.client.send_prompt(user_input)
    chat_manager.client.export_conversation(chat_manager.conversation_file)

    conversations = list_conversations()
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "messages": chat_manager.client.history,
        "conversations": conversations
    })

@app.post("/new_chat", response_class=RedirectResponse)
async def new_chat():
    chat_manager.new_chat()
    return RedirectResponse(url="/", status_code=303)

@app.get("/load_chat", response_class=RedirectResponse)
async def load_chat(filename: str = Query(...)):
    conversation_path = os.path.join(conversations_dir, filename)
    chat_manager.reset(conversation_path)
    return RedirectResponse(url="/", status_code=303)

@app.get("/settings", response_class=HTMLResponse)
async def settings(request: Request):
    conversations = list_conversations()
    settings_data = load_settings()
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "conversations": conversations,
        "settings": settings_data
    })
