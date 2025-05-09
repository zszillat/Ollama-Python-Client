"""
./ollama/exceptions.py
Custom exceptions for the Ollama client.
"""


class OllamaError(Exception):
    """Base exception for all Ollama-related errors."""
    pass


class OllamaRequestError(OllamaError):
    """Error when making a request to the Ollama API."""
    pass


class OllamaResponseError(OllamaError):
    """Error in the response from the Ollama API."""
    pass


class OllamaModelNotFoundError(OllamaError):
    """Error when a requested model is not found."""
    pass


class OllamaConnectionError(OllamaError):
    """Error connecting to the Ollama API."""
    pass