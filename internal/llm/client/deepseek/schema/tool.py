# llm/client/deepseek/schema/tool.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Literal, Union



@dataclass
class Function:
    description: str
    name: str
    parameters: Dict[str, Any]      # JSON Schema!
    strict: bool = False            # if strict, will ensure json output


@dataclass
class Tool:
    function: Function
    type: Literal["function"] = "function"


@dataclass
class NamedToolChoice:
    function: Dict[str, str]                # {"name": "func name"}
    type: Literal["function"] = "function"


ToolChoice = Union[Literal["none", "auto", "required"], NamedToolChoice]
