"""
Utility functions for the Ollama client.
./ollama/utils.py
"""
import os
import json
import base64
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Any, BinaryIO, Optional


def calculate_sha256(file_path: Union[str, Path]) -> str:
    """
    Calculate the SHA256 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: SHA256 hash in the format "sha256:digest"
    """
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # Read the file in chunks to avoid loading large files into memory
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
            
    return f"sha256:{sha256_hash.hexdigest()}"


def encode_image(image_path: Union[str, Path]) -> str:
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


def encode_file_content(file_path: Union[str, Path]) -> str:
    """
    Read and encode file content for inclusion in prompts.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: File content with markup for beginning and end
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    
    # For text files, read as text
    if ext in [".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".csv", ".xml", ".yaml", ".yml"]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    # For binary files, encode as base64
    else:
        with open(file_path, "rb") as f:
            content = base64.b64encode(f.read()).decode("utf-8")
            
    filename = path.name
    return f"<file name=\"{filename}\">\n{content}\n</file>"


def serialize_conversation(model: str, conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Serialize a conversation for export.
    
    Args:
        model: Name of the model
        conversation: List of conversation messages
        
    Returns:
        Dict: Serialized conversation
    """
    return {
        "model": model,
        "version": 1,  # For future compatibility
        "timestamp": str(datetime.now().isoformat()),
        "messages": conversation
    }


def deserialize_conversation(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Deserialize a conversation from imported data.
    
    Args:
        data: Serialized conversation data
        
    Returns:
        List[Dict]: Conversation messages
    """
    if not isinstance(data, dict):
        raise ValueError("Invalid conversation data format")
        
    if "version" not in data or "messages" not in data:
        raise ValueError("Invalid conversation format: missing required fields")
        
    return data.get("messages", [])


def format_file_prompt(file_path: Union[str, Path], description: Optional[str] = None) -> str:
    """
    Format a file for inclusion in a prompt.
    
    Args:
        file_path: Path to the file
        description: Optional description of the file
        
    Returns:
        str: Formatted file prompt
    """
    content = encode_file_content(file_path)
    if description:
        return f"{description}\n\n{content}"
    return content


def format_token_usage(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format token usage statistics from a response.
    
    Args:
        response: Response from the API
        
    Returns:
        Dict: Token usage information
    """
    return {
        "prompt_tokens": response.get("prompt_eval_count", 0),
        "completion_tokens": response.get("eval_count", 0),
        "total_tokens": (response.get("prompt_eval_count", 0) or 0) + (response.get("eval_count", 0) or 0)
    }


def format_duration(nanoseconds: Optional[int]) -> Optional[float]:
    """
    Format duration from nanoseconds to seconds.
    
    Args:
        nanoseconds: Duration in nanoseconds
        
    Returns:
        float: Duration in seconds or None if input is None
    """
    if nanoseconds is None:
        return None
    return nanoseconds / 1_000_000_000  # Convert to seconds