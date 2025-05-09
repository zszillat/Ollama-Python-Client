"""
./ollama/client.py
Main client implementation for interacting with the Ollama API.
"""
from typing import Dict, List, Union, Optional, Any, Iterator, Callable, AsyncIterator, Tuple
import os
import json
import base64
import httpx
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from .models import (
    Message, 
    ChatResponse, 
    GenerateResponse, 
    EmbedResponse, 
    ModelInfo,
    RunningModel,
    OllamaModel,
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
from .streaming import (
    StreamProcessor, 
    AsyncStreamProcessor,
    StreamHandler
)


class Ollama:
    """
    Client for interacting with the Ollama API.
    
    Provides methods for all Ollama API endpoints, including support for both
    streaming and non-streaming responses.
    
    Args:
        host: Base URL for the Ollama API. Defaults to http://localhost:11434.
        timeout: Timeout for API requests in seconds.
    
    Examples:
        >>> from ollama import Ollama
        >>> client = Ollama()
        >>> response = client.generate("llama3.2", "Why is the sky blue?", stream=False)
        >>> print(response.response)
    """
    
    def __init__(
        self, 
        host: str = None, 
        timeout: float = 60.0,
    ):
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        if not self.host.startswith(("http://", "https://")):
            self.host = f"http://{self.host}"
        self.host = self.host.rstrip('/')
        
        self.api_base = f"{self.host}/api"
        self.timeout = timeout
        self._conversations = {}  # Store conversation history
        self._http_client = httpx.Client(timeout=timeout)
    
    def __del__(self):
        if hasattr(self, '_http_client') and self._http_client:
            self._http_client.close()
    
    def _request(
        self, 
        method: str, 
        endpoint: str, 
        data: Dict = None, 
        params: Dict = None,
        stream: bool = False,
        files: Dict = None,
    ) -> Union[Dict, Iterator[Dict]]:
        """
        Send a request to the Ollama API.
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters
            stream: Whether to stream the response
            files: Files to upload
            
        Returns:
            Dict or Iterator[Dict]: API response
            
        Raises:
            OllamaError: If the request fails
        """
        url = f"{self.api_base}/{endpoint.lstrip('/')}"
        
        try:
            # Make the request
            if method.lower() == "get":
                response = self._http_client.get(url, params=params)
            elif method.lower() == "post":
                response = self._http_client.post(
                    url, json=data, params=params, files=files
                )
            elif method.lower() == "delete":
                response = self._http_client.delete(url, json=data, params=params)
            elif method.lower() == "head":
                response = self._http_client.head(url, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            if response.status_code == 204 or response.status_code == 201:
                return {}
                
            if stream:
                # Process as a stream
                return StreamProcessor(response)
            
            # Return the JSON response
            return response.json()
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise OllamaModelNotFoundError(f"Model not found: {e}")
            error_message = str(e)
            try:
                error_data = e.response.json()
                if "error" in error_data:
                    error_message = error_data["error"]
            except:
                pass
            raise OllamaRequestError(f"HTTP error: {error_message}")
            
        except httpx.RequestError as e:
            raise OllamaConnectionError(f"Connection error: {e}")
            
        except Exception as e:
            raise OllamaError(f"Unexpected error: {e}")

    @asynccontextmanager
    async def _async_client(self):
        """Context manager for async HTTP client."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            yield client

    async def _async_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Dict = None, 
        params: Dict = None,
        stream: bool = False
    ) -> Union[Dict, AsyncIterator[Dict]]:
        """
        Send an asynchronous request to the Ollama API.
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters
            stream: Whether to stream the response
            
        Returns:
            Dict or AsyncIterator[Dict]: API response
            
        Raises:
            OllamaError: If the request fails
        """
        url = f"{self.api_base}/{endpoint.lstrip('/')}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                if method.lower() == "get":
                    response = await client.get(url, params=params)
                elif method.lower() == "post":
                    response = await client.post(
                        url, json=data, params=params
                    )
                elif method.lower() == "delete":
                    response = await client.delete(url, json=data, params=params)
                elif method.lower() == "head":
                    response = await client.head(url, params=params)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                
                if response.status_code == 204 or response.status_code == 201:
                    return {}
                    
                if stream:
                    return AsyncStreamProcessor(response)
                
                # Convert to dictionary explicitly to avoid any await issues
                try:
                    # Try to get JSON directly first
                    response_text = response.text
                    import json
                    response_json = json.loads(response_text)
                    return response_json
                except Exception:
                    # Fall back to the httpx json method if needed
                    try:
                        json_method = response.json()
                        # Check if json_method is already a dict (not awaitable)
                        if isinstance(json_method, dict):
                            return json_method
                        # Otherwise, assume it's awaitable
                        response_json = await json_method
                        return response_json
                    except Exception as e:
                        raise OllamaError(f"Failed to parse JSON response: {e}")
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise OllamaModelNotFoundError(f"Model not found: {e}")
            error_message = str(e)
            try:
                error_data = await e.response.json()
                if "error" in error_data:
                    error_message = error_data["error"]
            except:
                pass
            raise OllamaRequestError(f"HTTP error: {error_message}")
            
        except httpx.RequestError as e:
            raise OllamaConnectionError(f"Connection error: {e}")
            
        except Exception as e:
            raise OllamaError(f"Unexpected error: {e}")

    def _encode_image(self, image_path: str) -> str:
        """
        Encode an image file to base64.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: Base64-encoded image data
        """
        with open(image_path, "rb") as f:
            image_data = f.read()
            return base64.b64encode(image_data).decode("utf-8")

    def _prepare_images(self, images: List[Union[str, bytes]]) -> List[str]:
        """
        Prepare images for inclusion in a request.
        
        Args:
            images: List of image paths or bytes data
            
        Returns:
            List[str]: List of base64-encoded image data
        """
        encoded_images = []
        for img in images:
            if isinstance(img, str):
                encoded_images.append(self._encode_image(img))
            elif isinstance(img, bytes):
                encoded_images.append(base64.b64encode(img).decode("utf-8"))
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
        return encoded_images

    def _prepare_chat_messages(
        self, 
        messages: List[Dict],
        system: Optional[str] = None,
        images: Optional[List[Union[str, bytes]]] = None,
    ) -> List[Dict]:
        """
        Prepare chat messages for a request.
        
        Args:
            messages: List of message dictionaries
            system: Optional system message
            images: Optional list of images for multimodal models
            
        Returns:
            List[Dict]: Processed message list ready for the API
        """
        processed_messages = []
        
        # Add system message if provided
        if system:
            processed_messages.append({"role": "system", "content": system})
            
        # Process existing messages
        for msg in messages:
            # Convert Message objects to dicts if needed
            if isinstance(msg, Message):
                msg = msg.to_dict()
                
            # Ensure required fields
            if "role" not in msg:
                raise ValueError("Message missing required 'role' field")
            if "content" not in msg:
                raise ValueError("Message missing required 'content' field")
                
            processed_messages.append(msg)
            
        # Add images to the last user message if provided
        if images and processed_messages:
            for i in range(len(processed_messages) - 1, -1, -1):
                if processed_messages[i]["role"] == "user":
                    processed_messages[i]["images"] = self._prepare_images(images)
                    break
        
        return processed_messages

    def generate(
        self, 
        model: str,
        prompt: str,
        system: Optional[str] = None,
        template: Optional[str] = None,
        format: Optional[Union[str, Dict]] = None,
        options: Optional[Dict[str, Any]] = None,
        images: Optional[List[Union[str, bytes]]] = None,
        raw: bool = False,
        stream: bool = True,
        keep_alive: Optional[str] = None,
    ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """
        Generate a completion for a given prompt.
        
        Args:
            model: Name of the model to use
            prompt: The prompt to generate a response for
            system: System message to use
            template: Custom prompt template
            format: Output format (e.g., "json")
            options: Additional model parameters
            images: List of image paths or bytes data for multimodal models
            raw: If true, no formatting will be applied to the prompt
            stream: Whether to stream the response (default: True)
            keep_alive: How long to keep the model loaded (e.g., "5m")
            
        Returns:
            GenerateResponse or Iterator[GenerateResponse]: The generated response
            
        Examples:
            >>> # Streaming (default)
            >>> for chunk in client.generate("llama3.2", "Why is the sky blue?"):
            >>>     print(chunk.response, end="")
            >>>     if chunk.done:
            >>>         print(f"\nTokens: {chunk.eval_count}")
            
            >>> # Non-streaming
            >>> response = client.generate("llama3.2", "Why is the sky blue?", stream=False)
            >>> print(response.response)
        """
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "raw": raw,
        }
        
        if system:
            data["system"] = system
        if template:
            data["template"] = template
        if format:
            data["format"] = format
        if options:
            data["options"] = options
        if keep_alive:
            data["keep_alive"] = keep_alive
        if images:
            data["images"] = self._prepare_images(images)
        
        if stream:
            stream_iterator = self._request("post", "generate", data=data, stream=True)
            return map(GenerateResponse.from_dict, stream_iterator)
        else:
            response = self._request("post", "generate", data=data)
            return GenerateResponse.from_dict(response)

    async def agenerate(
        self, 
        model: str,
        prompt: str,
        system: Optional[str] = None,
        template: Optional[str] = None,
        format: Optional[Union[str, Dict]] = None,
        options: Optional[Dict[str, Any]] = None,
        images: Optional[List[Union[str, bytes]]] = None,
        raw: bool = False,
        stream: bool = True,
        keep_alive: Optional[str] = None,
    ) -> Union[GenerateResponse, AsyncIterator[GenerateResponse]]:
        """
        Asynchronously generate a completion for a given prompt.
        
        Args:
            model: Name of the model to use
            prompt: The prompt to generate a response for
            system: System message to use
            template: Custom prompt template
            format: Output format (e.g., "json")
            options: Additional model parameters
            images: List of image paths or bytes data for multimodal models
            raw: If true, no formatting will be applied to the prompt
            stream: Whether to stream the response (default: True)
            keep_alive: How long to keep the model loaded (e.g., "5m")
            
        Returns:
            GenerateResponse or AsyncIterator[GenerateResponse]: The generated response
            
        Examples:
            >>> # Streaming (default)
            >>> async for chunk in client.agenerate("llama3.2", "Why is the sky blue?"):
            >>>     print(chunk.response, end="")
            >>>     if chunk.done:
            >>>         print(f"\nTokens: {chunk.eval_count}")
            
            >>> # Non-streaming
            >>> response = await client.agenerate("llama3.2", "Why is the sky blue?", stream=False)
            >>> print(response.response)
        """
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "raw": raw,
        }
        
        if system:
            data["system"] = system
        if template:
            data["template"] = template
        if format:
            data["format"] = format
        if options:
            data["options"] = options
        if keep_alive:
            data["keep_alive"] = keep_alive
        if images:
            data["images"] = self._prepare_images(images)
        
        if stream:
            stream_iterator = await self._async_request("post", "generate", data=data, stream=True)
            
            async def process_stream():
                async for chunk in stream_iterator:
                    yield GenerateResponse.from_dict(chunk)
                    
            return process_stream()
        else:
            response = await self._async_request("post", "generate", data=data)
            return GenerateResponse.from_dict(response)

    def chat(
        self,
        model: str,
        messages: List[Union[Dict, Message]],
        system: Optional[str] = None,
        format: Optional[Union[str, Dict]] = None,
        options: Optional[Dict[str, Any]] = None,
        images: Optional[List[Union[str, bytes]]] = None,
        stream: bool = True,
        keep_alive: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
    ) -> Union[ChatResponse, Iterator[ChatResponse]]:
        """
        Generate the next message in a chat conversation.
        
        Args:
            model: Name of the model to use
            messages: List of previous messages in the conversation
            system: System message to use
            format: Output format (e.g., "json")
            options: Additional model parameters
            images: List of image paths or bytes data for multimodal models
            stream: Whether to stream the response (default: True)
            keep_alive: How long to keep the model loaded (e.g., "5m")
            tools: List of tools available for the model to use
            
        Returns:
            ChatResponse or Iterator[ChatResponse]: The chat response
            
        Examples:
            >>> # Streaming (default)
            >>> messages = [{"role": "user", "content": "Hello, how are you?"}]
            >>> for chunk in client.chat("llama3.2", messages):
            >>>     print(chunk.message.content, end="")
            
            >>> # Non-streaming
            >>> messages = [{"role": "user", "content": "Hello, how are you?"}]
            >>> response = client.chat("llama3.2", messages, stream=False)
            >>> print(response.message.content)
        """
        # Save conversation ID based on model
        conv_id = f"{model}-{id(messages)}"
        if conv_id not in self._conversations:
            self._conversations[conv_id] = []
        
        # Process messages
        messages = self._prepare_chat_messages(messages, system, images)
        
        # Update our conversation history
        for msg in messages:
            if not any(existing_msg == msg for existing_msg in self._conversations[conv_id]):
                self._conversations[conv_id].append(msg)
        
        data = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        
        if format:
            data["format"] = format
        if options:
            data["options"] = options
        if keep_alive:
            data["keep_alive"] = keep_alive
        if tools:
            data["tools"] = tools
        
        if stream:
            stream_iterator = self._request("post", "chat", data=data, stream=True)
            
            def process_responses():
                final_response = None
                for chunk in stream_iterator:
                    response = ChatResponse.from_dict(chunk)
                    if response.done:
                        final_response = response
                    else:
                        yield response
                
                # After streaming completes, update conversation with the response
                if final_response and final_response.message:
                    self._conversations[conv_id].append(final_response.message.to_dict())
                    yield final_response
            
            return process_responses()
        else:
            response = self._request("post", "chat", data=data)
            response_obj = ChatResponse.from_dict(response)
            
            # Update conversation with the response
            if response_obj.message:
                self._conversations[conv_id].append(response_obj.message.to_dict())
                
            return response_obj

    async def achat(
        self,
        model: str,
        messages: List[Union[Dict, Message]],
        system: Optional[str] = None,
        format: Optional[Union[str, Dict]] = None,
        options: Optional[Dict[str, Any]] = None,
        images: Optional[List[Union[str, bytes]]] = None,
        stream: bool = True,
        keep_alive: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
    ) -> Union[ChatResponse, AsyncIterator[ChatResponse]]:
        """
        Asynchronously generate the next message in a chat conversation.
        
        Args:
            model: Name of the model to use
            messages: List of previous messages in the conversation
            system: System message to use
            format: Output format (e.g., "json")
            options: Additional model parameters
            images: List of image paths or bytes data for multimodal models
            stream: Whether to stream the response (default: True)
            keep_alive: How long to keep the model loaded (e.g., "5m")
            tools: List of tools available for the model to use
            
        Returns:
            ChatResponse or AsyncIterator[ChatResponse]: The chat response
        """
        # Save conversation ID based on model
        conv_id = f"{model}-{id(messages)}"
        if conv_id not in self._conversations:
            self._conversations[conv_id] = []
        
        # Process messages
        messages = self._prepare_chat_messages(messages, system, images)
        
        # Update our conversation history
        for msg in messages:
            if not any(existing_msg == msg for existing_msg in self._conversations[conv_id]):
                self._conversations[conv_id].append(msg)
        
        data = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        
        if format:
            data["format"] = format
        if options:
            data["options"] = options
        if keep_alive:
            data["keep_alive"] = keep_alive
        if tools:
            data["tools"] = tools
        
        if stream:
            stream_iterator = await self._async_request("post", "chat", data=data, stream=True)
            
            async def process_responses():
                async for chunk in stream_iterator:
                    response = ChatResponse.from_dict(chunk)
                    if response.done and response.message:
                        self._conversations[conv_id].append(response.message.to_dict())
                    yield response
            
            return process_responses()
        else:
            response = await self._async_request("post", "chat", data=data)
            response_obj = ChatResponse.from_dict(response)
            
            # Update conversation with the response
            if response_obj.message:
                self._conversations[conv_id].append(response_obj.message.to_dict())
                
            return response_obj

    def create(
        self,
        model: str,
        from_model: Optional[str] = None,
        files: Optional[Dict[str, str]] = None,
        adapters: Optional[Dict[str, str]] = None,
        template: Optional[str] = None,
        license: Optional[Union[str, List[str]]] = None,
        system: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict]] = None,
        stream: bool = True,
        quantize: Optional[str] = None,
    ) -> Union[CreateModelResponse, Iterator[CreateModelResponse]]:
        """
        Create a model from another model, safetensors, or GGUF files.
        
        Args:
            model: Name of the model to create
            from_model: Name of an existing model to create from
            files: Dictionary of file names to SHA256 digests of blobs
            adapters: Dictionary of file names to SHA256 digests for LORA adapters
            template: Prompt template for the model
            license: License or licenses for the model
            system: System prompt for the model
            parameters: Dictionary of parameters for the model
            messages: List of message objects for conversation
            stream: Whether to stream the response
            quantize: Quantize a non-quantized model
            
        Returns:
            CreateModelResponse or Iterator[CreateModelResponse]: Creation status
        """
        data = {
            "model": model,
            "stream": stream,
        }
        
        if from_model:
            data["from"] = from_model
        if files:
            data["files"] = files
        if adapters:
            data["adapters"] = adapters
        if template:
            data["template"] = template
        if license:
            data["license"] = license
        if system:
            data["system"] = system
        if parameters:
            data["parameters"] = parameters
        if messages:
            data["messages"] = messages
        if quantize:
            data["quantize"] = quantize
        
        if stream:
            stream_iterator = self._request("post", "create", data=data, stream=True)
            return map(CreateModelResponse.from_dict, stream_iterator)
        else:
            response = self._request("post", "create", data=data)
            return CreateModelResponse.from_dict(response)

    async def acreate(
        self,
        model: str,
        from_model: Optional[str] = None,
        files: Optional[Dict[str, str]] = None,
        adapters: Optional[Dict[str, str]] = None,
        template: Optional[str] = None,
        license: Optional[Union[str, List[str]]] = None,
        system: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict]] = None,
        stream: bool = True,
        quantize: Optional[str] = None,
    ) -> Union[CreateModelResponse, AsyncIterator[CreateModelResponse]]:
        """
        Asynchronously create a model from another model, safetensors, or GGUF files.
        
        Args:
            model: Name of the model to create
            from_model: Name of an existing model to create from
            files: Dictionary of file names to SHA256 digests of blobs
            adapters: Dictionary of file names to SHA256 digests for LORA adapters
            template: Prompt template for the model
            license: License or licenses for the model
            system: System prompt for the model
            parameters: Dictionary of parameters for the model
            messages: List of message objects for conversation
            stream: Whether to stream the response
            quantize: Quantize a non-quantized model
            
        Returns:
            CreateModelResponse or AsyncIterator[CreateModelResponse]: Creation status
        """
        data = {
            "model": model,
            "stream": stream,
        }
        
        if from_model:
            data["from"] = from_model
        if files:
            data["files"] = files
        if adapters:
            data["adapters"] = adapters
        if template:
            data["template"] = template
        if license:
            data["license"] = license
        if system:
            data["system"] = system
        if parameters:
            data["parameters"] = parameters
        if messages:
            data["messages"] = messages
        if quantize:
            data["quantize"] = quantize
        
        if stream:
            stream_iterator = await self._async_request("post", "create", data=data, stream=True)
            
            async def process_stream():
                async for chunk in stream_iterator:
                    yield CreateModelResponse.from_dict(chunk)
                    
            return process_stream()
        else:
            response = await self._async_request("post", "create", data=data)
            return CreateModelResponse.from_dict(response)

    def list_models(self) -> List[OllamaModel]:
        """
        List all models available locally.
        
        Returns:
            List[OllamaModel]: List of models
        """
        response = self._request("get", "tags")
        models = response.get("models", [])
        return [OllamaModel.from_dict(model) for model in models]

    async def alist_models(self) -> List[OllamaModel]:
        """
        Asynchronously list all models available locally.
        
        Returns:
            List[OllamaModel]: List of models
        """
        response = await self._async_request("get", "tags")
        models = response.get("models", [])
        return [OllamaModel.from_dict(model) for model in models]

    def show(self, model: str, verbose: bool = False) -> ModelInfo:
        """
        Show information about a model.
        
        Args:
            model: Name of the model
            verbose: If true, returns full data for verbose response fields
            
        Returns:
            ModelInfo: Information about the model
            
        Raises:
            OllamaModelNotFoundError: If the model is not found
        """
        data = {"model": model}
        if verbose:
            data["verbose"] = True
            
        response = self._request("post", "show", data=data)
        return ModelInfo.from_dict(response)

    async def ashow(self, model: str, verbose: bool = False) -> ModelInfo:
        """
        Asynchronously show information about a model.
        
        Args:
            model: Name of the model
            verbose: If true, returns full data for verbose response fields
            
        Returns:
            ModelInfo: Information about the model
            
        Raises:
            OllamaModelNotFoundError: If the model is not found
        """
        data = {"model": model}
        if verbose:
            data["verbose"] = True
            
        response = await self._async_request("post", "show", data=data)
        return ModelInfo.from_dict(response)

    def copy(self, source: str, destination: str) -> bool:
        """
        Copy a model.
        
        Args:
            source: Source model name
            destination: Destination model name
            
        Returns:
            bool: Whether the operation was successful
            
        Raises:
            OllamaModelNotFoundError: If the source model is not found
        """
        data = {
            "source": source,
            "destination": destination,
        }
        
        try:
            self._request("post", "copy", data=data)
            return True
        except OllamaError:
            return False

    async def acopy(self, source: str, destination: str) -> bool:
        """
        Asynchronously copy a model.
        
        Args:
            source: Source model name
            destination: Destination model name
            
        Returns:
            bool: Whether the operation was successful
            
        Raises:
            OllamaModelNotFoundError: If the source model is not found
        """
        data = {
            "source": source,
            "destination": destination,
        }
        
        try:
            await self._async_request("post", "copy", data=data)
            return True
        except OllamaError:
            return False

    def delete(self, model: str) -> bool:
        """
        Delete a model.
        
        Args:
            model: Name of the model to delete
            
        Returns:
            bool: Whether the operation was successful
        """
        data = {"model": model}
        
        try:
            self._request("delete", "delete", data=data)
            return True
        except OllamaError:
            return False

    async def adelete(self, model: str) -> bool:
        """
        Asynchronously delete a model.
        
        Args:
            model: Name of the model to delete
            
        Returns:
            bool: Whether the operation was successful
        """
        data = {"model": model}
        
        try:
            await self._async_request("delete", "delete", data=data)
            return True
        except OllamaError:
            return False

    def pull(
        self, 
        model: str, 
        insecure: bool = False, 
        stream: bool = True
    ) -> Union[Dict[str, str], Iterator[Dict[str, Any]]]:
        """
        Pull a model from the Ollama library.
        
        Args:
            model: Name of the model to pull
            insecure: Allow insecure connections
            stream: Whether to stream the progress
            
        Returns:
            Dict or Iterator[Dict]: Pull status information
            
        Examples:
            >>> # Stream progress (default)
            >>> for status in client.pull("llama3.2"):
            >>>     print(f"Status: {status.get('status')}")
            
            >>> # Non-streaming
            >>> result = client.pull("llama3.2", stream=False)
            >>> print(f"Status: {result.get('status')}")
        """
        data = {
            "model": model,
            "insecure": insecure,
            "stream": stream,
        }
        
        if stream:
            return self._request("post", "pull", data=data, stream=True)
        else:
            return self._request("post", "pull", data=data)

    async def apull(
        self, 
        model: str, 
        insecure: bool = False, 
        stream: bool = True
    ) -> Union[Dict[str, str], AsyncIterator[Dict[str, Any]]]:
        """
        Asynchronously pull a model from the Ollama library.
        
        Args:
            model: Name of the model to pull
            insecure: Allow insecure connections
            stream: Whether to stream the progress
            
        Returns:
            Dict or AsyncIterator[Dict]: Pull status information
            
        Examples:
            >>> # Stream progress (default)
            >>> async for status in client.apull("llama3.2"):
            >>>     print(f"Status: {status.get('status')}")
            
            >>> # Non-streaming
            >>> result = await client.apull("llama3.2", stream=False)
            >>> print(f"Status: {result.get('status')}")
        """
        data = {
            "model": model,
            "insecure": insecure,
            "stream": stream,
        }
        
        if stream:
            return await self._async_request("post", "pull", data=data, stream=True)
        else:
            return await self._async_request("post", "pull", data=data)

    def push(
        self, 
        model: str, 
        insecure: bool = False, 
        stream: bool = True
    ) -> Union[Dict[str, str], Iterator[Dict[str, Any]]]:
        """
        Push a model to the Ollama library.
        
        Args:
            model: Name of the model to push in the form of <namespace>/<model>:<tag>
            insecure: Allow insecure connections
            stream: Whether to stream the progress
            
        Returns:
            Dict or Iterator[Dict]: Push status information
            
        Examples:
            >>> # Stream progress (default)
            >>> for status in client.push("myuser/mymodel:latest"):
            >>>     print(f"Status: {status.get('status')}")
        """
        data = {
            "model": model,
            "insecure": insecure,
            "stream": stream,
        }
        
        if stream:
            return self._request("post", "push", data=data, stream=True)
        else:
            return self._request("post", "push", data=data)

    async def apush(
        self, 
        model: str, 
        insecure: bool = False, 
        stream: bool = True
    ) -> Union[Dict[str, str], AsyncIterator[Dict[str, Any]]]:
        """
        Asynchronously push a model to the Ollama library.
        
        Args:
            model: Name of the model to push in the form of <namespace>/<model>:<tag>
            insecure: Allow insecure connections
            stream: Whether to stream the progress
            
        Returns:
            Dict or AsyncIterator[Dict]: Push status information
        """
        data = {
            "model": model,
            "insecure": insecure,
            "stream": stream,
        }
        
        if stream:
            return await self._async_request("post", "push", data=data, stream=True)
        else:
            return await self._async_request("post", "push", data=data)

    def embed(
        self,
        model: str,
        input: Union[str, List[str]],
        truncate: bool = True,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> EmbedResponse:
        """
        Generate embeddings from a model.
        
        Args:
            model: Name of model to generate embeddings from
            input: Text or list of text to generate embeddings for
            truncate: Whether to truncate the input to fit context length
            options: Additional model parameters
            keep_alive: Controls how long the model will stay loaded
            
        Returns:
            EmbedResponse: The generated embeddings
            
        Examples:
            >>> # Single input
            >>> result = client.embed("all-minilm", "Why is the sky blue?")
            >>> print(f"Embeddings shape: {len(result.embeddings[0])}")
            
            >>> # Multiple inputs
            >>> result = client.embed("all-minilm", ["Why is the sky blue?", "Hello world"])
            >>> print(f"Number of embeddings: {len(result.embeddings)}")
        """
        data = {
            "model": model,
            "input": input,
            "truncate": truncate,
        }
        
        if options:
            data["options"] = options
        if keep_alive:
            data["keep_alive"] = keep_alive
            
        response = self._request("post", "embed", data=data)
        return EmbedResponse.from_dict(response)

    async def aembed(
        self,
        model: str,
        input: Union[str, List[str]],
        truncate: bool = True,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> EmbedResponse:
        """
        Asynchronously generate embeddings from a model.
        
        Args:
            model: Name of model to generate embeddings from
            input: Text or list of text to generate embeddings for
            truncate: Whether to truncate the input to fit context length
            options: Additional model parameters
            keep_alive: Controls how long the model will stay loaded
            
        Returns:
            EmbedResponse: The generated embeddings
        """
        data = {
            "model": model,
            "input": input,
            "truncate": truncate,
        }
        
        if options:
            data["options"] = options
        if keep_alive:
            data["keep_alive"] = keep_alive
            
        response = await self._async_request("post", "embed", data=data)
        return EmbedResponse.from_dict(response)

    def list_running_models(self) -> List[RunningModel]:
        """
        List currently running models.
        
        Returns:
            List[RunningModel]: List of running models
        """
        response = self._request("get", "ps")
        models = response.get("models", [])
        return [RunningModel.from_dict(model) for model in models]

    async def alist_running_models(self) -> List[RunningModel]:
        """
        Asynchronously list currently running models.
        
        Returns:
            List[RunningModel]: List of running models
        """
        response = await self._async_request("get", "ps")
        models = response.get("models", [])
        return [RunningModel.from_dict(model) for model in models]

    def version(self) -> OllamaVersion:
        """
        Get Ollama version information.
        
        Returns:
            OllamaVersion: Version information
        """
        response = self._request("get", "version")
        return OllamaVersion.from_dict(response)

    async def aversion(self) -> OllamaVersion:
        """
        Asynchronously get Ollama version information.
        
        Returns:
            OllamaVersion: Version information
        """
        response = await self._async_request("get", "version")
        return OllamaVersion.from_dict(response)

    def blob_exists(self, digest: str) -> bool:
        """
        Check if a blob exists.
        
        Args:
            digest: SHA256 digest of the blob
            
        Returns:
            bool: Whether the blob exists
        """
        try:
            self._request("head", f"blobs/{digest}")
            return True
        except OllamaError:
            return False

    async def ablob_exists(self, digest: str) -> bool:
        """
        Asynchronously check if a blob exists.
        
        Args:
            digest: SHA256 digest of the blob
            
        Returns:
            bool: Whether the blob exists
        """
        try:
            await self._async_request("head", f"blobs/{digest}")
            return True
        except OllamaError:
            return False

    def push_blob(self, file_path: str, digest: str) -> BlobStatus:
        """
        Push a file to the Ollama server as a blob.
        
        Args:
            file_path: Path to the file
            digest: SHA256 digest of the file
            
        Returns:
            BlobStatus: Result of the operation
            
        Example:
            >>> status = client.push_blob("model.gguf", "sha256:123...")
            >>> print(f"Success: {status.success}")
        """
        try:
            with open(file_path, "rb") as f:
                files = {"file": (Path(file_path).name, f)}
                self._request("post", f"blobs/{digest}", files=files)
                
            return BlobStatus(success=True, error=None)
        except Exception as e:
            return BlobStatus(success=False, error=str(e))

    async def apush_blob(self, file_path: str, digest: str) -> BlobStatus:
        """
        Asynchronously push a file to the Ollama server as a blob.
        
        Args:
            file_path: Path to the file
            digest: SHA256 digest of the file
            
        Returns:
            BlobStatus: Result of the operation
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                with open(file_path, "rb") as f:
                    form = {"file": (Path(file_path).name, f.read())}
                    response = await client.post(
                        f"{self.api_base}/blobs/{digest}",
                        files=form
                    )
                    response.raise_for_status()
                    
            return BlobStatus(success=True, error=None)
        except Exception as e:
            return BlobStatus(success=False, error=str(e))

    def get_conversation_history(self, model: str, messages: List[Dict]) -> List[Dict]:
        """
        Get the conversation history for a model and messages.
        
        Args:
            model: Name of the model
            messages: Original messages list
            
        Returns:
            List[Dict]: Conversation history
        """
        conv_id = f"{model}-{id(messages)}"
        return self._conversations.get(conv_id, [])

    def clear_conversation_history(self, model: str, messages: List[Dict]) -> bool:
        """
        Clear the conversation history for a model and messages.
        
        Args:
            model: Name of the model
            messages: Original messages list
            
        Returns:
            bool: Whether the operation was successful
        """
        conv_id = f"{model}-{id(messages)}"
        if conv_id in self._conversations:
            self._conversations[conv_id] = []
            return True
        return False
    
    def with_stream_handler(
        self, 
        handler: Union[StreamHandler, Callable]
    ) -> "Ollama":
        """
        Set a stream handler for all streaming operations.
        
        Args:
            handler: Stream handler function or class
            
        Returns:
            Ollama: Self for chaining
            
        Example:
            >>> # Using a function
            >>> client = Ollama().with_stream_handler(lambda chunk: print(chunk.response))
            >>> client.generate("llama3.2", "Hello")  # Will print chunks automatically
            
            >>> # Using a class
            >>> class MyHandler(StreamHandler):
            >>>     def handle_chunk(self, chunk):
            >>>         print(f"Got chunk: {chunk.response}")
            >>> client = Ollama().with_stream_handler(MyHandler())
        """
        StreamProcessor.set_global_handler(handler)
        return self