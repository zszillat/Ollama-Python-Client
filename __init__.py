"""
Ollama Python Client Library - A comprehensive client for the Ollama API
"""

__version__ = "0.1.0"

from .client import Ollama
from .models import (
    Message,
    ChatResponse,
    GenerateResponse,
    EmbedResponse,
    ModelInfo,
    OllamaModel,
    RunningModel,
    CreateModelResponse,
    OllamaVersion,
    BlobStatus,
)
from .exceptions import (
    OllamaError,
    OllamaRequestError,
    OllamaResponseError,
    OllamaModelNotFoundError,
    OllamaConnectionError,
)
from .streaming import StreamHandler
from .modelfile import Modelfile, load_modelfile, parse_modelfile

# Convenience re-exports
from .utils import (
    calculate_sha256,
    encode_image,
    encode_file_content,
    format_file_prompt,
)