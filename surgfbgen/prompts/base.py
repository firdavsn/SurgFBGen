"""Base module for prompt management."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Set

class PromptTemplate:
    """Base class for prompt templates."""

    def __init__(
        self,
        template: str,
        name: str = None,
        description: str = "1.0",
        version: str = "1.0",
        metadata: Dict[str, Any] = None,
        parameters: Dict[str, Dict[str, Any]] = None,
    ) -> None:
        """Initialize a prompt template.

        Args:
            template: The prompt template string with placeholders
            name: Name of the prompt template
            description: Description of the template
            version: Version of the template
            metadata: Additional metadata
            parameters: Dictionary of parameter descriptions defining the expected
                placeholders, e.g. {"question": {"description": "The user question", "required": True},
                                 "context": {"description": "The retrieved context", "required": True}}
        """
        self.template = template
        self.name = name
        self.description = description
        self.version = version
        self.metadata = metadata or {}

        # Extract parameters from template if not provided
        if parameters is None:
            self.parameters = self._extract_parameters_from_template()
        else:
            self.parameters = parameters

    def _extract_parameters_from_template(self) -> Dict[str, Dict[str, Any]]:
        """Extract parameters from template string using regex.
        Find all formatter placeholders in the template.
        """
        placeholders = re.findall(r"\{([^}]+)\}", self.template)
        parameters = {}

        # Create a base parameter description for each placeholder
        for param in placeholders:
            # skip any format specifiers like {param:02d}
            param_name = param.split(":")[0]
            if param_name not in parameters:
                parameters[param_name] = {
                    "description": f"Parameter: {param_name}",
                    "required": True
                }

        return parameters

    def format(self, **kwargs) -> str:
        """Format the template with the given arguments."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required placeholder in prompt template: {e}")

    def get_required_parameters(self) -> List[str]:
        """Get list of required parameters."""
        return [
            param for param, details in self.parameters.items()
            if details.get("required", True)
        ]

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate if the provided parameters match the required ones."""
        required_params = set(self.get_required_parameters())
        provided_params = set(params.keys())

        return required_params.issubset(provided_params)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the template to a dictionary."""
        return {
            "name": self.name,
            "template": self.template,
            "description": self.description,
            "version": self.version,
            "metadata": self.metadata,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create a template from a dictionary."""
        return cls(
            name=data.get("name"),
            template=data.get("template"),
            description=data.get("description"),
            version=data.get("version"),
            metadata=data.get("metadata"),
            parameters=data.get("parameters"),
        )

    @classmethod
    def from_file(cls, file_path: str) -> "PromptTemplate":
        """Load a template from a file.

        Supports both text files (raw template) and JSON files (with metadata).
        """
        with open(file_path, "r") as f:
            content = f.read()

        # Check if it's a JSON file
        if file_path.endswith(".json"):
            try:
                data = json.loads(content)
                return cls.from_dict(data)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON file: {file_path}")

        # Otherwise treat as raw template
        name = os.path.basename(file_path).split(".")[0]
        return cls(template=content, name=name)


class PromptLibrary:
    """Library for managing prompt templates."""

    def __init__(self) -> None:
        """Initialize the prompt library."""
        self._prompts: Dict[str, PromptTemplate] = {}

    def add(self, prompt: PromptTemplate) -> None:
        """Add a prompt template to the library."""
        if not prompt.name:
            raise ValueError("Prompt template must have a name")
        self._prompts[prompt.name] = prompt

    def get(self, name: str) -> Optional[PromptTemplate]:
        """Get a prompt template by name."""
        return self._prompts.get(name)

    def remove(self, name: str) -> None:
        """Remove a prompt template from the library."""
        if name in self._prompts:
            del self._prompts[name]

    def list(self) -> List[Dict[str, Any]]:
        """List all prompt templates in the library."""
        return [
            {
                "name": name,
                "description": prompt.description,
                "version": prompt.version,
                "parameters": {
                    k: v.get("description", "")
                    for k, v in prompt.parameters.items()
                } if hasattr(prompt, "parameters") else {},
            }
            for name, prompt in self._prompts.items()
        ]

    def load_from_directory(self, directory: str) -> None:
        """Load all prompt templates from a directory."""
        # Loads .txt files as raw templates and .json files as templates with metadata.
        if not os.path.isdir(directory):
            raise ValueError(f"Directory not found: {directory}")

        for filename in os.listdir(directory):
            if filename.endswith((".txt", ".json")):
                file_path = os.path.join(directory, filename)
                prompt_template = PromptTemplate.from_file(file_path)
                self.add(prompt_template)

    def save_to_directory(self, directory: str) -> None:
        """Save all prompt templates to a directory."""
        os.makedirs(directory, exist_ok=True)

        for name, prompt in self._prompts.items():
            # Save JSON files for templates with metadata
            file_path = os.path.join(directory, f"{name}.json")
            with open(file_path, "w") as f:
                json.dump(prompt.to_dict(), f, indent=2)

# Global instance of the prompt library
prompt_library = PromptLibrary()