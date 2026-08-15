from dataclasses import dataclass
from typing import Literal
from .block import ContentBlock


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: list[ContentBlock]


@dataclass
class SystemMessage(Message):
    def __init__(self, content: list[ContentBlock]):
        super().__init__("system", content)


@dataclass
class UserMessage(Message):
    def __init__(self, content: list[ContentBlock]):
        super().__init__("user", content)


@dataclass
class AssistantMessage(Message):
    def __init__(self, content: list[ContentBlock]):
        super().__init__("assistant", content)


@dataclass
class ToolMessage(Message):
    def __init__(self, content: list[ContentBlock]):
        super().__init__("tool", content)
