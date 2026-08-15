# llm/client/deepseek/schema/message.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TypeAlias


@dataclass
class SystemMessage:
    content: str
    role: str = "system"


@dataclass
class UserMessage:
    content: str
    role: str = "user"


@dataclass
class AssistantMessage:
    content: Optional[str] = None
    role: str = "assistant"

    # (Beta) used for prefix filling
    # prefix: bool
    # reasoning_content: string


@dataclass
class ToolMessage:
    content: str
    tool_call_id: str
    role: str = "tool"


Message: TypeAlias = (
    SystemMessage
    | UserMessage
    | AssistantMessage
    | ToolMessage
)
