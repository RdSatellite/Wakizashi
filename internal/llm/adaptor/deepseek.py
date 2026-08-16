# llm/adaptor/deepseek.py

from __future__ import annotations

import json
from typing import List

from ..entity import (
    Message,
    AssistantMessage,

    ContentBlock,
    TextBlock,
    ReasoningBlock,
    ToolCallBlock,
    ToolResultBlock
)

from ..client.deepseek import DeepseekClient
from ..client.deepseek.schema import (
    AssistantMessage as DsAssistantMessage,
    ChatCompletionResponse,
    FunctionCallDelta,
    SystemMessage as DsSystemMessage,
    ToolCallDelta,
    ToolMessage as DsToolMessage,
    UserMessage as DsUserMessage,
)


class StreamAccumulator:
    """Accumulates streaming ChatCompletionResponse chunks into an entity AssistantMessage."""

    def __init__(self):
        self._reasoning_parts: list[str] = []
        self._content_parts: list[str] = []
        self._tool_calls: dict[int, _ToolCallAccum] = {}

    def accumulate(self, chunk: ChatCompletionResponse) -> AssistantMessage | None:
        choice = chunk.choices[0]
        delta = choice.delta

        if delta.reasoning_content:
            self._reasoning_parts.append(delta.reasoning_content)

        if delta.content:
            self._content_parts.append(delta.content)

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                if tc_delta.index not in self._tool_calls:
                    self._tool_calls[tc_delta.index] = _ToolCallAccum()
                tc_acc = self._tool_calls[tc_delta.index]
                if tc_delta.id:
                    tc_acc.id = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        tc_acc.name = tc_delta.function.name
                    if tc_delta.function.arguments:
                        tc_acc.arguments_parts.append(tc_delta.function.arguments)

        if choice.finish_reason:
            return self._build_message()

        return None

    def _build_message(self) -> AssistantMessage:
        blocks: list[ContentBlock] = []

        if self._reasoning_parts:
            blocks.append(ReasoningBlock(reasoning="".join(self._reasoning_parts)))

        if self._content_parts:
            blocks.append(TextBlock(text="".join(self._content_parts)))

        for index in sorted(self._tool_calls):
            tc_acc = self._tool_calls[index]
            args_str = "".join(tc_acc.arguments_parts)
            try:
                arguments = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                arguments = {"_raw": args_str}
            blocks.append(ToolCallBlock(id=tc_acc.id, name=tc_acc.name, arguments=arguments))

        return AssistantMessage(content=blocks)


class _ToolCallAccum:
    def __init__(self):
        self.id: str = ""
        self.name: str = ""
        self.arguments_parts: list[str] = []


class DeepseekAdaptor:

    def to_deepseek_messages(self, messages: List[Message]) -> List[
        DsSystemMessage | DsUserMessage | DsAssistantMessage | DsToolMessage
    ]:
        return [self._convert_message(msg) for msg in messages]

    def create_accumulator(self) -> StreamAccumulator:
        return StreamAccumulator()

    def _convert_message(self, message: Message):
        match message:
            case Message(role="system"):
                return DsSystemMessage(content=self._extract_text(message.content))
            case Message(role="user"):
                return DsUserMessage(content=self._extract_text(message.content))
            case Message(role="assistant"):
                text = self._extract_text(message.content)
                reasoning = self._extract_reasoning(message.content)
                tool_calls = self._extract_tool_calls(message.content)
                return DsAssistantMessage(
                    content=text if text else "",
                    reasoning_content=reasoning if (tool_calls and reasoning) else None,
                    tool_calls=tool_calls if tool_calls else None,
                )
            case Message(role="tool"):
                return DsToolMessage(
                    content=self._extract_tool_output(message.content),
                    tool_call_id=self._extract_tool_call_id(message.content),
                )
            case _:
                raise ValueError(f"Unknown message role: {message.role}")

    @staticmethod
    def _extract_text(blocks: list[ContentBlock]) -> str:
        parts = []
        for block in blocks:
            if isinstance(block, TextBlock):
                parts.append(block.text)
        return "\n".join(parts)

    @staticmethod
    def _extract_reasoning(blocks: list[ContentBlock]) -> str:
        parts = []
        for block in blocks:
            if isinstance(block, ReasoningBlock):
                parts.append(block.reasoning)
        return "".join(parts)

    @staticmethod
    def _extract_tool_calls(blocks: list[ContentBlock]) -> list[ToolCallDelta]: 
        result = []
        for block in blocks:
            if isinstance(block, ToolCallBlock):
                result.append(ToolCallDelta(
                    index=len(result),
                    id=block.id or None,
                    function=FunctionCallDelta(
                        name=block.name,
                        arguments=json.dumps(block.arguments, ensure_ascii=False),
                    ),
                ))
        return result

    @staticmethod
    def _extract_tool_output(blocks: list[ContentBlock]) -> str:
        parts = []
        for block in blocks:
            if isinstance(block, ToolResultBlock):
                output = block.output
                parts.append(output if isinstance(output, str) else json.dumps(output))
        return "\n".join(parts) if parts else ""

    @staticmethod
    def _extract_tool_call_id(blocks: list[ContentBlock]) -> str:
        for block in blocks:
            if isinstance(block, ToolResultBlock) and block.tool_call_id:
                return block.tool_call_id
        return ""
