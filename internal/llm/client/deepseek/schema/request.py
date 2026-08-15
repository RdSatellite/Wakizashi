# llm/client/deepseek/schema/request.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Literal

from .message import Message
from .tool import Tool, ToolChoice


@dataclass
class ChatCompletionRequest:
    # Commentted default is platform's default settings
    # Pass None to trigger

    messages: list[Message]
    model: Literal["deepseek-v4-flash", "deepseek-v4-pro"]

    thinking: Optional[ThinkingType] = None
    reasoning_effort: Optional[Literal["low", "high", "max"]] = None    # Default: high
    stream: Optional[bool] = True                                       # Default: False

    temperature: Optional[float] = None    # Default: 1.0, temp <= 2
    top_p: Optional[float] = None          # Default: 1.0

    tools: Optional[list[Tool]] = None
    tool_choice: Optional[ToolChoice] = None

    @staticmethod
    def builder() -> ChatCompletionRequestBuilder:
        return ChatCompletionRequestBuilder()


class ChatCompletionRequestBuilder:
    def __init__(self):
        self._messages: Optional[List[Message]] = None
        self._model: Optional[Literal["deepseek-v4-flash", "deepseek-v4-pro"]] = None

        self._thinking: Optional[ThinkingType] = ThinkingType()
        self._reasoning_effort: Optional[Literal["low", "high", "max"]] = None
        self._stream: bool = True
        self._temperature: Optional[float] = None
        self._top_p: Optional[float] = None
        self._tools: Optional[List[Tool]] = None
        self._tool_choice: Optional[ToolChoice] = None

    def set_messages(self, messages: List[Message]) -> ChatCompletionRequestBuilder:
        self._messages = messages
        return self

    def set_model(self, model: Literal["deepseek-v4-flash", "deepseek-v4-pro"]) -> ChatCompletionRequestBuilder:
        self._model = model
        return self

    # --- Thinking --- #

    def enable_thinking(self, effort: Literal["low", "high", "max"] = "high") -> ChatCompletionRequestBuilder:
        self._thinking = ThinkingType(type="enabled")
        self._reasoning_effort = effort
        return self

    def disable_thinking(self) -> ChatCompletionRequestBuilder:
        self._thinking = ThinkingType(type="disabled")
        self._reasoning_effort = None
        return self

    # --- Streaming --- #

    def set_stream(self, stream: bool) -> ChatCompletionRequestBuilder:
        self._stream = stream
        return self

    # --- Sampling --- #

    def set_temperature(self, temperature: float) -> ChatCompletionRequestBuilder:
        if temperature > 2:
            raise ValueError(f"temperature cannot be greater than 2, current: {temperature}")
        self._temperature = temperature
        return self

    def set_top_p(self, top_p: float) -> ChatCompletionRequestBuilder:
        self._top_p = top_p
        return self

    # --- Tool Calling --- #

    def set_tools(self, tools: List[Tool], choice: ToolChoice = "auto") -> ChatCompletionRequestBuilder:
        self._tools = tools
        self._tool_choice = choice
        return self


    def build(self) -> ChatCompletionRequest:
        if not self._messages:
            raise ValueError("Failed to build ChatCompletionRequest: messages cannot be None")
        if not self._model:
            raise ValueError("Failed to build ChatCompletionRequest: model cannot be None")

        return ChatCompletionRequest(
            messages=self._messages,
            model=self._model,
            thinking=self._thinking,
            reasoning_effort=self._reasoning_effort,
            stream=self._stream,
            temperature=self._temperature,
            top_p=self._top_p,
            tools=self._tools,
            tool_choice=self._tool_choice,
        )


@dataclass
class ThinkingType:
    type: Literal["enabled", "disabled"] = "enabled"
