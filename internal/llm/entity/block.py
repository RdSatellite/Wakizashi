from dataclasses import dataclass
from typing import Any, TypeAlias

"""Agent's final response to human"""
@dataclass
class TextBlock:
    text: str

"""Reasoning texts in <think>, supported by some models"""
@dataclass
class ReasoningBlock:
    reasoning: str

"""Agent starts a tool calling, and creates a ToolCallBlock"""
@dataclass
class ToolCallBlock:
    name: str
    arguments: dict[str, Any]
    id: str = ""

"""Tool calling's result"""
@dataclass
class ToolResultBlock:
    output: Any
    tool_call_id: str = ""


ContentBlock: TypeAlias = (
    TextBlock
    | ReasoningBlock
    | ToolCallBlock
    | ToolResultBlock
)

