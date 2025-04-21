import requests
from typing import List, Optional, Dict, Any


class Options:
    def __init__(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        num_predict: Optional[int] = None,
        stop: Optional[List[str]] = None,
        repeat_last_n: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        mirostat: Optional[int] = None,
        mirostat_tau: Optional[float] = None,
        mirostat_eta: Optional[float] = None,
        num_ctx: Optional[int] = None
    ):
        # All optional generation parameters supported by the Ollama API
        # These map directly to the POST payload for /api/generate
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_predict = num_predict
        self.stop = stop
        self.repeat_last_n = repeat_last_n
        self.repeat_penalty = repeat_penalty
        self.seed = seed
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.mirostat = mirostat
        self.mirostat_tau = mirostat_tau
        self.mirostat_eta = mirostat_eta
        self.num_ctx = num_ctx
        self.stream = False  # stream is not supported

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class Ollama:
    def __init__(self, base_url: str, model: str):
        """
        Parameters:
        - base_url (str): Base URL of the Ollama server (e.g., http://localhost:11434)
        - model (str): Model name to use (must be pre-installed)
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.last_request = None

    def send_prompt(self, prompt: str, options: Optional[Options] = None) -> str:
        """
        Send a prompt to the Ollama API using the specified model and optional generation parameters.

        Parameters:
        - prompt (str): The text prompt to send
        - options (Options, optional): Generation parameters

        Returns:
        - str: The generated response text only

        Also saves the full API JSON response to `self.last_request`.
        """
        payload = {
            "model": self.model,
            "prompt": prompt
        }

        if options:
            payload.update(options.to_dict())
        else:
            payload["stream"] = False

        response = requests.post(f"{self.base_url}/api/generate", json=payload)
        response.raise_for_status()
        self.last_request = response.json()
        return self.last_request.get("response", "")