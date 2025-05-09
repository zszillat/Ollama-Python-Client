"""
./ollama/modelfile.py
Utilities for working with Ollama modelfiles.
"""
from typing import Dict, List, Union, Any, Optional
from pathlib import Path


class Modelfile:
    """
    Class for creating and managing Ollama Modelfiles.
    
    This provides a programmatic way to build Modelfiles to be used
    with the Ollama API's create endpoint.
    
    Example:
        >>> modelfile = Modelfile.from_model("llama3.2")
        >>> modelfile.set_system("You are a helpful assistant.")
        >>> modelfile.set_parameter("temperature", 0.7)
        >>> with open("Modelfile", "w") as f:
        >>>     f.write(str(modelfile))
    """
    
    def __init__(self, from_value: str):
        """
        Initialize a Modelfile with a base model or file.
        
        Args:
            from_value: Base model name or path to a model file
        """
        self.from_value = from_value
        self.parameters = {}
        self.system = None
        self.template = None
        self.license = None
        self.adapters = []
        self.messages = []
        
    @classmethod
    def from_model(cls, model_name: str) -> "Modelfile":
        """
        Create a Modelfile from an existing model.
        
        Args:
            model_name: Name of the model to use as base
            
        Returns:
            Modelfile: New Modelfile instance
        """
        return cls(model_name)
        
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "Modelfile":
        """
        Create a Modelfile from a model file.
        
        Args:
            file_path: Path to the model file (GGUF or safetensors)
            
        Returns:
            Modelfile: New Modelfile instance
        """
        return cls(str(file_path))
        
    def set_parameter(self, name: str, value: Union[str, int, float, bool]) -> "Modelfile":
        """
        Set a parameter for the model.
        
        Args:
            name: Parameter name
            value: Parameter value
            
        Returns:
            Modelfile: Self for chaining
        """
        self.parameters[name] = value
        return self
        
    def set_system(self, system_prompt: str) -> "Modelfile":
        """
        Set the system prompt for the model.
        
        Args:
            system_prompt: System prompt text
            
        Returns:
            Modelfile: Self for chaining
        """
        self.system = system_prompt
        return self
        
    def set_template(self, template: str) -> "Modelfile":
        """
        Set the prompt template for the model.
        
        Args:
            template: Template text
            
        Returns:
            Modelfile: Self for chaining
        """
        self.template = template
        return self
        
    def set_license(self, license_text: Union[str, List[str]]) -> "Modelfile":
        """
        Set the license for the model.
        
        Args:
            license_text: License text or list of license texts
            
        Returns:
            Modelfile: Self for chaining
        """
        self.license = license_text
        return self
        
    def add_adapter(self, adapter_path: Union[str, Path]) -> "Modelfile":
        """
        Add a LORA adapter to the model.
        
        Args:
            adapter_path: Path to the adapter file
            
        Returns:
            Modelfile: Self for chaining
        """
        self.adapters.append(str(adapter_path))
        return self
        
    def add_message(self, role: str, content: str) -> "Modelfile":
        """
        Add a message to the conversation history.
        
        Args:
            role: Message role (system, user, assistant)
            content: Message content
            
        Returns:
            Modelfile: Self for chaining
        """
        self.messages.append({"role": role, "content": content})
        return self
        
    def _format_value(self, value: Any) -> str:
        """
        Format a value for inclusion in the Modelfile.
        
        Args:
            value: Value to format
            
        Returns:
            str: Formatted value
        """
        if isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            if "\n" in value:
                return f'"""\n{value}\n"""'
            else:
                return f'"{value}"'
        else:
            return str(value)
        
    def __str__(self) -> str:
        """
        Convert the Modelfile to its string representation.
        
        Returns:
            str: Modelfile content
        """
        lines = []
        
        # FROM instruction (required)
        lines.append(f"FROM {self.from_value}")
        
        # PARAMETER instructions
        for name, value in self.parameters.items():
            lines.append(f"PARAMETER {name} {self._format_value(value)}")
        
        # TEMPLATE instruction
        if self.template:
            lines.append(f"TEMPLATE {self._format_value(self.template)}")
        
        # SYSTEM instruction
        if self.system:
            lines.append(f"SYSTEM {self._format_value(self.system)}")
        
        # ADAPTER instructions
        for adapter in self.adapters:
            lines.append(f"ADAPTER {adapter}")
        
        # LICENSE instruction
        if self.license:
            if isinstance(self.license, list):
                license_text = "\n".join(self.license)
            else:
                license_text = self.license
            lines.append(f"LICENSE {self._format_value(license_text)}")
        
        # MESSAGE instructions
        for message in self.messages:
            lines.append(f"MESSAGE {message['role']} {self._format_value(message['content'])}")
        
        return "\n".join(lines)
    
    def to_file(self, file_path: Union[str, Path]) -> None:
        """
        Write the Modelfile to a file.
        
        Args:
            file_path: Path to write the file to
        """
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(str(self))


def parse_modelfile(content: str) -> Modelfile:
    """
    Parse a Modelfile from its content.
    
    Args:
        content: Content of the Modelfile
        
    Returns:
        Modelfile: Parsed Modelfile instance
        
    Raises:
        ValueError: If the Modelfile is invalid or missing required instructions
    """
    lines = content.split("\n")
    from_value = None
    parameters = {}
    system = None
    template = None
    license = None
    adapters = []
    messages = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines and comments
        if not line or line.startswith("#"):
            i += 1
            continue
        
        # Parse instruction
        parts = line.split(" ", 1)
        if len(parts) < 2:
            i += 1
            continue
            
        instruction, args = parts
        instruction = instruction.upper()
        
        # Handle multi-line values
        if '"""' in args:
            # Find the end of the multi-line value
            value_lines = []
            # Skip the opening quote
            value_start = args.find('"""') + 3
            if value_start < len(args):
                value_lines.append(args[value_start:])
            
            i += 1
            while i < len(lines):
                if '"""' in lines[i]:
                    # End of multi-line value
                    end_idx = lines[i].find('"""')
                    if end_idx >= 0:
                        value_lines.append(lines[i][:end_idx])
                    break
                value_lines.append(lines[i])
                i += 1
                
            value = "\n".join(value_lines)
        else:
            # Single line value
            value = args.strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
        
        # Process instructions
        if instruction == "FROM":
            from_value = value
        elif instruction == "PARAMETER":
            param_parts = args.split(" ", 1)
            if len(param_parts) == 2:
                param_name, param_value = param_parts
                # Convert parameter value to appropriate type
                if param_value.lower() == "true":
                    parameters[param_name] = True
                elif param_value.lower() == "false":
                    parameters[param_name] = False
                elif param_value.isdigit():
                    parameters[param_name] = int(param_value)
                elif param_value.replace(".", "", 1).isdigit():
                    parameters[param_name] = float(param_value)
                else:
                    # Remove quotes if present
                    if param_value.startswith('"') and param_value.endswith('"'):
                        param_value = param_value[1:-1]
                    parameters[param_name] = param_value
        elif instruction == "SYSTEM":
            system = value
        elif instruction == "TEMPLATE":
            template = value
        elif instruction == "LICENSE":
            license = value
        elif instruction == "ADAPTER":
            adapters.append(value)
        elif instruction == "MESSAGE":
            message_parts = args.split(" ", 1)
            if len(message_parts) == 2:
                role, content = message_parts
                # Remove quotes if present
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]
                messages.append({"role": role, "content": content})
        
        i += 1
    
    if not from_value:
        raise ValueError("Modelfile must contain a FROM instruction")
    
    modelfile = Modelfile(from_value)
    for name, value in parameters.items():
        modelfile.set_parameter(name, value)
    if system:
        modelfile.set_system(system)
    if template:
        modelfile.set_template(template)
    if license:
        modelfile.set_license(license)
    for adapter in adapters:
        modelfile.add_adapter(adapter)
    for message in messages:
        modelfile.add_message(message["role"], message["content"])
    
    return modelfile


def load_modelfile(file_path: Union[str, Path]) -> Modelfile:
    """
    Load a Modelfile from a file.
    
    Args:
        file_path: Path to the Modelfile
        
    Returns:
        Modelfile: Loaded Modelfile instance
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return parse_modelfile(content)