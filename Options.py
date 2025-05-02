from typing import Any, Dict, List, Optional

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
        # Set generation parameters
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
        # Never stream
        self.stream = False

    def to_dict(self) -> Dict[str, Any]:
        # Include only set values
        return {k: v for k, v in self.__dict__.items() if v is not None}
