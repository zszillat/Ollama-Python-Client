import requests
import json
from typing import Any, Dict, List, Optional, Union
from .Options import Options  # Import Options from the same package

class Ollama:
    def __init__(
        self,
        base_url_or_file: str,
        model: Optional[str] = None
    ):
        # Load saved session or start a new one
        if model is None:
            with open(base_url_or_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.base_url = data['base_url'].rstrip("/")
            self.model = data['model']
            self.history = data.get('history', [])
            self.last_request = data.get('last_request')
        else:
            self.base_url = base_url_or_file.rstrip("/")
            self.model = model
            self.history = []
            self.last_request = None

    def add_message(self, role: str, content: str) -> None:
        # Record chat message
        self.history.append({"role": role, "content": content})

    def clear_history(self) -> None:
        # Reset history
        self.history = []

    def send_prompt(
        self,
        prompt: str,
        options: Optional[Options] = None,
        suffix: Optional[str] = None,
        system: Optional[str] = None,
        template: Optional[str] = None,
        raw: Optional[bool] = None,
        keep_alive: Optional[str] = None,
        images: Optional[List[str]] = None,
        format: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> str:
        # Add user input
        self.add_message("user", prompt)

        # Build request
        payload: Dict[str, Any] = {"model": self.model, "messages": self.history}
        for key, val in [("suffix", suffix), ("system", system), ("template", template),
                         ("raw", raw), ("keep_alive", keep_alive), ("images", images), ("format", format)]:
            if val is not None:
                payload[key] = val

        if options:
            opts = options.to_dict()
            if opts:
                payload.update(opts)
        payload["stream"] = False

        # Call chat API
        r = requests.post(f"{self.base_url}/api/chat", json=payload)
        r.raise_for_status()
        self.last_request = r.json()

        # Get text reply
        msg = self.last_request.get("message", {})
        resp = msg.get("content") if isinstance(msg, dict) else self.last_request.get("response", "")

        # Save reply
        self.add_message("assistant", resp)
        return resp

    def export_conversation(self, file_path: str) -> None:
        # Save session
        data = {
            "base_url": self.base_url,
            "model": self.model,
            "history": self.history,
            "last_request": self.last_request
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
