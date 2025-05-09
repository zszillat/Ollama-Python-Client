"""
./ollama/models.py
Data models for the Ollama API.
"""
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Message:
    """A message in a conversation."""
    role: str
    content: str
    images: Optional[List[str]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "role": self.role,
            "content": self.content
        }
        
        if self.images:
            result["images"] = self.images
            
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
            
        return result
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary representation."""
        return cls(
            role=data.get("role", ""),
            content=data.get("content", ""),
            images=data.get("images"),
            tool_calls=data.get("tool_calls")
        )


@dataclass
class GenerateResponse:
    """Response from the generate endpoint."""
    model: str
    created_at: Optional[str] = None
    response: str = ""
    done: bool = False
    done_reason: Optional[str] = None
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerateResponse":
        """Create from dictionary representation."""
        return cls(
            model=data.get("model", ""),
            created_at=data.get("created_at"),
            response=data.get("response", ""),
            done=data.get("done", False),
            done_reason=data.get("done_reason"),
            context=data.get("context"),
            total_duration=data.get("total_duration"),
            load_duration=data.get("load_duration"),
            prompt_eval_count=data.get("prompt_eval_count"),
            prompt_eval_duration=data.get("prompt_eval_duration"),
            eval_count=data.get("eval_count"),
            eval_duration=data.get("eval_duration")
        )


@dataclass
class ChatResponse:
    """Response from the chat endpoint."""
    model: str
    created_at: Optional[str] = None
    message: Optional[Message] = None
    done: bool = False
    done_reason: Optional[str] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatResponse":
        """Create from dictionary representation."""
        message = None
        if "message" in data:
            message = Message.from_dict(data["message"])
            
        return cls(
            model=data.get("model", ""),
            created_at=data.get("created_at"),
            message=message,
            done=data.get("done", False),
            done_reason=data.get("done_reason"),
            total_duration=data.get("total_duration"),
            load_duration=data.get("load_duration"),
            prompt_eval_count=data.get("prompt_eval_count"),
            prompt_eval_duration=data.get("prompt_eval_duration"),
            eval_count=data.get("eval_count"),
            eval_duration=data.get("eval_duration")
        )


@dataclass
class EmbedResponse:
    """Response from the embed endpoint."""
    model: str
    embeddings: List[List[float]]
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbedResponse":
        """Create from dictionary representation."""
        # Handle both 'embedding' (singular) and 'embeddings' (plural) keys
        embeddings = data.get("embeddings", [])
        if not embeddings and "embedding" in data:
            embeddings = [data["embedding"]]
            
        return cls(
            model=data.get("model", ""),
            embeddings=embeddings,
            total_duration=data.get("total_duration"),
            load_duration=data.get("load_duration"),
            prompt_eval_count=data.get("prompt_eval_count")
        )


@dataclass
class ModelDetails:
    """Model details information."""
    format: Optional[str] = None
    family: Optional[str] = None
    families: Optional[List[str]] = None
    parameter_size: Optional[str] = None
    quantization_level: Optional[str] = None
    parent_model: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelDetails":
        """Create from dictionary representation."""
        return cls(
            format=data.get("format"),
            family=data.get("family"),
            families=data.get("families"),
            parameter_size=data.get("parameter_size"),
            quantization_level=data.get("quantization_level"),
            parent_model=data.get("parent_model")
        )


@dataclass
class OllamaModel:
    """Information about a model."""
    name: str
    modified_at: str
    size: int
    digest: str
    details: Optional[ModelDetails] = None
    
    @property
    def modified_datetime(self) -> datetime:
        """Convert modified_at string to datetime object."""
        return datetime.fromisoformat(self.modified_at.replace("Z", "+00:00"))
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OllamaModel":
        """Create from dictionary representation."""
        details = None
        if "details" in data:
            details = ModelDetails.from_dict(data["details"])
            
        return cls(
            name=data.get("name", ""),
            modified_at=data.get("modified_at", ""),
            size=data.get("size", 0),
            digest=data.get("digest", ""),
            details=details
        )


@dataclass
class RunningModel:
    """Information about a running model."""
    name: str
    model: str
    size: int
    digest: str
    details: Optional[ModelDetails] = None
    expires_at: Optional[str] = None
    size_vram: Optional[int] = None
    
    @property
    def expires_datetime(self) -> Optional[datetime]:
        """Convert expires_at string to datetime object."""
        if self.expires_at:
            return datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        return None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunningModel":
        """Create from dictionary representation."""
        details = None
        if "details" in data:
            details = ModelDetails.from_dict(data["details"])
            
        return cls(
            name=data.get("name", ""),
            model=data.get("model", ""),
            size=data.get("size", 0),
            digest=data.get("digest", ""),
            details=details,
            expires_at=data.get("expires_at"),
            size_vram=data.get("size_vram")
        )


@dataclass
class ModelInfo:
    """Detailed information about a model."""
    modelfile: Optional[str] = None
    parameters: Optional[str] = None
    template: Optional[str] = None
    details: Optional[ModelDetails] = None
    model_info: Optional[Dict[str, Any]] = None
    capabilities: Optional[List[str]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        """Create from dictionary representation."""
        details = None
        if "details" in data:
            details = ModelDetails.from_dict(data["details"])
            
        return cls(
            modelfile=data.get("modelfile"),
            parameters=data.get("parameters"),
            template=data.get("template"),
            details=details,
            model_info=data.get("model_info"),
            capabilities=data.get("capabilities")
        )


@dataclass
class CreateModelResponse:
    """Response from the create model endpoint."""
    status: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CreateModelResponse":
        """Create from dictionary representation."""
        return cls(
            status=data.get("status", "")
        )


@dataclass
class OllamaVersion:
    """Version information."""
    version: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OllamaVersion":
        """Create from dictionary representation."""
        return cls(
            version=data.get("version", "")
        )


@dataclass
class BlobStatus:
    """Status of a blob operation."""
    success: bool
    error: Optional[str] = None