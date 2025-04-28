# app.py
from fastapi import FastAPI, Request, Form, Query, Body, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import glob
import json
import requests

from ollama_client import Options, Ollama

# Paths
current_dir = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))
static_dir = os.path.join(current_dir, "static")
conversations_dir = os.path.join(current_dir, "conversations")
settings_file = os.path.join(current_dir, "settings.json")
title_prompt = (
    "based on the conversation so far, what would be a good title for it?"
    " Respond with a conversation title only;"
    " use _ in place of a space;"
    " keep the title under 10 words"
)

# Config - Load from settings.json
settings_data = {}
with open(settings_file, 'r') as f:
    settings_data = json.load(f)

# Config
BASE_URL = settings_data.get('base_url', 'http://10.0.0.1:11434')
DEFAULT_MODEL = settings_data.get('manageModels', {}).get('modelInstalled', ['qwen2.5-coder:1.5b'])[0]

def get_installed_models(base_url):
    try:
        response = requests.get(f"{base_url}/api/tags")
        response.raise_for_status()
        tags = response.json()
        return [model['name'] for model in tags.get('models', [])]
    except Exception as e:
        print(f"Failed to fetch models: {e}")
        return []

app = FastAPI()

deleted_dir = os.path.join(current_dir, "deleted")
os.makedirs(deleted_dir, exist_ok=True)
os.makedirs(conversations_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

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
    raw_conversations = list_conversations()
    formatted_conversations = [os.path.splitext(name)[0].replace("_", " ") for name in raw_conversations]
    settings_data = load_settings()
    presets = settings_data.get('manageModels', {}).get('modelPresets', [])
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "messages": chat_manager.client.history,
            "conversations": formatted_conversations,
            "presets": presets,
        },
    )

@app.post("/delete_chat")
async def delete_chat(filename: str = Form(...)):
    safe_filename = filename.replace(" ", "_") + ".json"
    file_path = os.path.join(conversations_dir, safe_filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Conversation not found")

    currently_loaded_file = os.path.basename(chat_manager.conversation_file)
    deleting_current_chat = safe_filename == currently_loaded_file

    deleted_path = os.path.join(deleted_dir, safe_filename)
    os.rename(file_path, deleted_path)

    if deleting_current_chat:
        chat_manager.new_chat()
        return RedirectResponse(url="/", status_code=303)

    return RedirectResponse(url="/", status_code=303)

@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, user_input: str = Form(...)):
    chat_manager.client.send_prompt(user_input)
    chat_manager.client.export_conversation(chat_manager.conversation_file)

    filename = os.path.basename(chat_manager.conversation_file)
    if filename.startswith("chat_"):
        title_response = chat_manager.client.send_prompt(title_prompt)
        clean_title = "".join(
            c for c in title_response.strip() if c.isalnum() or c in (" ", "_", "-")
        ).strip().replace(" ", "_")
        new_filename = f"{clean_title}.json"
        new_file_path = os.path.join(os.path.dirname(chat_manager.conversation_file), new_filename)
        chat_manager.client.history = chat_manager.client.history[:-2]
        chat_manager.client.export_conversation(new_file_path)
        if os.path.exists(chat_manager.conversation_file):
            os.remove(chat_manager.conversation_file)
        chat_manager.conversation_file = new_file_path

    conversations = list_conversations()
    formatted_conversations = [os.path.splitext(name)[0].replace("_", " ") for name in conversations]
    settings_data = load_settings()
    presets = settings_data.get('manageModels', {}).get('modelPresets', [])
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "messages": chat_manager.client.history,
            "conversations": formatted_conversations,
            "presets": presets,
        },
    )

@app.post("/new_chat", response_class=RedirectResponse)
async def new_chat():
    chat_manager.new_chat()
    return RedirectResponse(url="/", status_code=303)

@app.get("/load_chat", response_class=RedirectResponse)
async def load_chat(filename: str = Query(...)):
    safe_filename = filename.replace(" ", "_") + ".json"
    conversation_path = os.path.join(conversations_dir, safe_filename)

    if not os.path.exists(conversation_path):
        return RedirectResponse(url="/", status_code=303)

    chat_manager.reset(conversation_path)
    return RedirectResponse(url="/", status_code=303)

@app.get("/settings", response_class=HTMLResponse)
async def settings(request: Request):
    raw_conversations = list_conversations()
    formatted_conversations = [os.path.splitext(name)[0].replace("_", " ") for name in raw_conversations]
    settings_data = load_settings()
    installed_models = get_installed_models(settings_data.get('base_url', 'http://10.0.0.1:11434'))
    return templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "conversations": formatted_conversations,
            "settings": settings_data,
            "installed_models": installed_models,
        },
    )
