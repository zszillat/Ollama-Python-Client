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
