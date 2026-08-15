"""轻量检视：DeepSeek 请求序列化 / 响应反序列化是否如预期"""

from dataclasses import asdict
from types import SimpleNamespace

from internal.llm.client.deepseek.schema import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Delta,
    UserMessage,
)
from internal.llm.client.deepseek.schema.tool import Function, Tool

print("=" * 70)
print("1) 请求序列化：asdict(ChatCompletionRequest)")
print("=" * 70)

req = ChatCompletionRequest(
    messages=[UserMessage(content="设计和艺术有什么区别？")],
    model="deepseek-v4-flash",
)
print(asdict(req))

print()
print("=" * 70)
print("2) 过滤 None 后的 kwargs（与 stream_chat 里完全相同的逻辑）")
print("=" * 70)

kwargs = asdict(req)
kwargs = {k: v for k, v in kwargs.items() if v is not None}
for k, v in kwargs.items():
    print(f"  {k:16} = {v!r}   ({type(v).__name__})")

print()
print("  关键检查：")
print(f"    kwargs 里已含 'stream' -> {'stream' in kwargs}")
print("    => create(**kwargs, stream=True) 会重复传 stream，触发 TypeError")

print()
print("=" * 70)
print("2b) builder 路径（默认 thinking=ThinkingType()）序列化")
print("=" * 70)

req2 = (
    ChatCompletionRequest.builder()
    .set_messages([UserMessage(content="hi")])
    .set_model("deepseek-v4-flash")
    .build()
)
print(asdict(req2))

print()
print("=" * 70)
print("3) tool 序列化：asdict(Tool)")
print("=" * 70)

tool = Tool(function=Function(
    name="get_weather",
    description="查询天气",
    parameters={"type": "object", "properties": {}},
))
print(asdict(tool))

print()
print("=" * 70)
print("4) 反序列化：模拟流式 chunk -> ChatCompletionResponse")
print("=" * 70)


def make_chunk(content=None, reasoning=None, finish=None, role=None, index=0):
    delta = SimpleNamespace(content=content, reasoning_content=reasoning, role=role)
    choice = SimpleNamespace(delta=delta, index=index, finish_reason=finish)
    return SimpleNamespace(
        id="chatcmpl-123",
        choices=[choice],
        created=1234567890,
        model="deepseek-v4-flash",
        object="chat.completion.chunk",
        system_fingerprint="fp_abc",
    )


# 与 stream_chat 里解析逻辑一致
chunks = [
    make_chunk(reasoning="让我想想…", role="assistant"),
    make_chunk(content="艺术是人类……"),
    make_chunk(finish="stop"),
]
for i, chunk in enumerate(chunks):
    raw_choice = chunk.choices[0]
    raw_delta = raw_choice.delta
    delta = Delta(
        content=raw_delta.content,
        reasoning_content=getattr(raw_delta, "reasoning_content", None),
        role=raw_delta.role,
    )
    choice = Choice(
        delta=delta,
        index=raw_choice.index,
        finish_reason=raw_choice.finish_reason,
    )
    resp = ChatCompletionResponse(
        id=chunk.id,
        choices=[choice],
        created=chunk.created,
        model=chunk.model,
        object=chunk.object,
        system_fingerprint=getattr(chunk, "system_fingerprint", None),
    )
    print(f"  chunk#{i}: {asdict(resp)}")
