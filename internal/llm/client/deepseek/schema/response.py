# llm/client/deepseek/schema/response.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, List


@dataclass
class FunctionCallDelta:
    name: Optional[str] = None
    arguments: Optional[str] = None


@dataclass
class ToolCallDelta:
    index: int
    id: Optional[str] = None
    type: Literal["function"] = "function"
    function: Optional[FunctionCallDelta] = None


@dataclass
class Delta:
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    role: Optional[str] = None
    tool_calls: Optional[List[ToolCallDelta]] = None


@dataclass
class Choice:
    delta: Delta
    index: int
    finish_reason: Literal[
        "stop",                         # Normal stop
        "length",                       # Length limit exceed
        "content_filter",               # Triggered filter strategy
        "insufficient_system_resource", # Deepseek's problem
        "tool_calls",                   # Model finished calling tools
    ]

    # logprobs: object


@dataclass
class ChatCompletionResponse:
    id: str
    choices: List[Choice]
    created: int
    model: str
    object: str
    system_fingerprint: str
