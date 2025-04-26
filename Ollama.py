import requests
import json
from typing import Any, Dict, List, Optional, Union

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
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        mirostat: Optional[int] = None,
        mirostat_tau: Optional[float] = None,
        mirostat_eta: Optional[float] = None,
        num_ctx: Optional[int] = None,
        # extra generation controls
        num_keep: Optional[int] = None,
        min_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        penalize_newline: Optional[bool] = None,
        numa: Optional[bool] = None,
        num_batch: Optional[int] = None,
        num_gpu: Optional[int] = None,
        main_gpu: Optional[int] = None,
        low_vram: Optional[bool] = None,
        vocab_only: Optional[bool] = None,
        use_mmap: Optional[bool] = None,
        use_mlock: Optional[bool] = None,
        num_thread: Optional[int] = None,
    ):
        # set generation parameters
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_predict = num_predict
        self.stop = stop
        self.repeat_last_n = repeat_last_n
        self.repeat_penalty = repeat_penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.seed = seed
        self.mirostat = mirostat
        self.mirostat_tau = mirostat_tau
        self.mirostat_eta = mirostat_eta
        self.num_ctx = num_ctx
        # advanced options
        self.num_keep = num_keep
        self.min_p = min_p
        self.typical_p = typical_p
        self.penalize_newline = penalize_newline
        self.numa = numa
        self.num_batch = num_batch
        self.num_gpu = num_gpu
        self.main_gpu = main_gpu
        self.low_vram = low_vram
        self.vocab_only = vocab_only
        self.use_mmap = use_mmap
        self.use_mlock = use_mlock
        self.num_thread = num_thread
        # never stream
        self.stream = False

    def to_dict(self) -> Dict[str, Any]:
        # include only set values
        return {k: v for k, v in self.__dict__.items() if v is not None}

class Ollama:
    def __init__(
        self,
        base_url_or_file: str,
        model: Optional[str] = None
    ):
        # load saved or start new
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
        # record chat message
        self.history.append({"role": role, "content": content})

    def clear_history(self) -> None:
        # reset history
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
        # add user input
        self.add_message("user", prompt)

        # build request
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

        # call chat API
        r = requests.post(f"{self.base_url}/api/chat", json=payload)
        r.raise_for_status()
        self.last_request = r.json()

        # get text reply
        msg = self.last_request.get("message", {})
        resp = msg.get("content") if isinstance(msg, dict) else self.last_request.get("response", "")

        # save reply
        self.add_message("assistant", resp)
        return resp

    def export_conversation(self, file_path: str) -> None:
        # save session
        data = {"base_url": self.base_url, "model": self.model,
                "history": self.history, "last_request": self.last_request}
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
