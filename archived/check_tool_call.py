"""轻量验证：流式工具调用 (tool_calls) 的 schema + 解析"""
import asyncio
from dataclasses import asdict
from types import SimpleNamespace

from internal.llm.client.deepseek.client import DeepseekClient
from internal.llm.client.deepseek.schema import ChatCompletionRequest, UserMessage


def tc(index, id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index,
        id=id,
        type="function" if id is not None else None,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def chunk(tool_calls=None, finish=None):
    delta = SimpleNamespace(
        content=None,
        reasoning_content=None,
        role="assistant",
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(delta=delta, index=0, finish_reason=finish)
    return SimpleNamespace(
        id="chatcmpl-1",
        choices=[choice],
        created=1,
        model="deepseek-v4-flash",
        object="chat.completion.chunk",
        system_fingerprint=None,
    )


CHUNKS = [
    chunk(tool_calls=[tc(0, id="call_1", name="get_weather", arguments="")]),
    chunk(tool_calls=[tc(0, arguments='{"location": "Beijing"}')]),
    chunk(tool_calls=[tc(1, id="call_2", name="get_time", arguments="")]),
    chunk(finish="tool_calls"),
]


class AsyncChunkIterator:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


async def fake_create(**kwargs):
    return AsyncChunkIterator(CHUNKS)


async def main():
    client = DeepseekClient(api_key="x")
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )

    req = ChatCompletionRequest(
        messages=[UserMessage(content="北京天气怎么样")],
        model="deepseek-v4-flash",
    )

    print("流式工具调用解析结果：")
    async for resp in client.stream_chat(req):
        print("  " + str(asdict(resp)))


asyncio.run(main())
