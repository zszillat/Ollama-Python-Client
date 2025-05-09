"""
./ollama/streaming.py
Streaming response handler for the Ollama API.
"""
from typing import Dict, Any, Iterator, AsyncIterator, Callable, Protocol, Union, Optional
import json
import httpx
from abc import ABC, abstractmethod


class StreamHandler(ABC):
    """Abstract base class for stream handlers."""
    
    @abstractmethod
    def handle_chunk(self, chunk: Dict[str, Any]) -> None:
        """Handle a chunk of data from the stream."""
        pass


class StreamProcessor:
    """
    Process streamed responses from the Ollama API.
    
    This class handles parsing JSON lines from streaming responses
    and applying the optional stream handler.
    """
    
    _global_handler: Optional[Union[StreamHandler, Callable]] = None
    
    def __init__(self, response: httpx.Response):
        """Initialize with an HTTP response object."""
        self.response = response
        
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Process the stream line by line, yielding JSON objects."""
        try:
            for line in self.response.iter_lines():
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    
                    # Apply global handler if set
                    if StreamProcessor._global_handler:
                        if isinstance(StreamProcessor._global_handler, StreamHandler):
                            StreamProcessor._global_handler.handle_chunk(data)
                        else:
                            StreamProcessor._global_handler(data)
                            
                    yield data
                except json.JSONDecodeError:
                    # Skip non-JSON lines
                    continue
        except Exception as e:
            raise OllamaError(f"Error processing stream: {e}")

    
    @classmethod
    def set_global_handler(cls, handler: Union[StreamHandler, Callable]) -> None:
        """Set a global handler for all stream processors."""
        cls._global_handler = handler
        
    @classmethod
    def get_global_handler(cls) -> Optional[Union[StreamHandler, Callable]]:
        """Get the current global handler."""
        return cls._global_handler
        
    @classmethod
    def clear_global_handler(cls) -> None:
        """Clear the global handler."""
        cls._global_handler = None


class AsyncStreamProcessor:
    """
    Process streamed responses from the Ollama API asynchronously.
    
    This class handles parsing JSON lines from streaming responses
    and applying the optional stream handler.
    """
    
    def __init__(self, response: httpx.Response):
        """Initialize with an HTTP response object."""
        self.response = response
        
    async def __aiter__(self) -> AsyncIterator[Dict[str, Any]]:
        """Process the stream line by line, yielding JSON objects."""
        try:
            async for line in self.response.aiter_lines():
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    
                    # Apply global handler if set
                    if StreamProcessor._global_handler:
                        if isinstance(StreamProcessor._global_handler, StreamHandler):
                            StreamProcessor._global_handler.handle_chunk(data)
                        else:
                            StreamProcessor._global_handler(data)
                            
                    yield data
                except json.JSONDecodeError:
                    # Skip non-JSON lines
                    continue
        except Exception as e:
            raise OllamaError(f"Error processing async stream: {e}")