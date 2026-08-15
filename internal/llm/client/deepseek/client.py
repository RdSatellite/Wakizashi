# llm/client/deepseek/client.py

import os
import asyncio
from typing import AsyncGenerator, Union, List
from dataclasses import asdict

from .schema import (
    Choice,
    Delta,
    FunctionCallDelta,
    ToolCallDelta,
    ChatCompletionRequest,
    ChatCompletionResponse
)

from openai import AsyncOpenAI


class DeepseekClient:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

    async def stream_chat(
        self, 
        req: ChatCompletionRequest
    ) -> AsyncGenerator[ChatCompletionResponse, None]:

        kwargs = asdict(req)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        rsp = await self.client.chat.completions.create(**kwargs) # stream=True/False is already contained in req

        async for chunk in rsp:
            if not chunk.choices:
                continue

            raw_choice = chunk.choices[0]
            raw_delta = raw_choice.delta

            raw_tool_calls = getattr(raw_delta, "tool_calls", None)
            tool_calls = None
            if raw_tool_calls:
                tool_calls = [
                    ToolCallDelta(
                        index=tc.index,
                        id=getattr(tc, "id", None),
                        function=FunctionCallDelta(
                            name=getattr(tc.function, "name", None),
                            arguments=getattr(tc.function, "arguments", None),
                        )
                        if tc.function
                        else None,
                    )
                    for tc in raw_tool_calls
                ]

            delta = Delta(
                content=raw_delta.content,
                reasoning_content=getattr(raw_delta, "reasoning_content", None),
                role=raw_delta.role,
                tool_calls=tool_calls,
            )

            choice = Choice(
                delta=delta,
                index=raw_choice.index,
                finish_reason=raw_choice.finish_reason
            )

            response = ChatCompletionResponse(
                id=chunk.id,
                choices=[choice],
                created=chunk.created,
                model=chunk.model,
                object=chunk.object,
                system_fingerprint=getattr(chunk, "system_fingerprint", None)
            )

            yield response


if __name__ == "__main__":
    api_key = os.getenv("DEEPSEEK_API_KEY")
    client = DeepseekClient(api_key=api_key)

    from .schema.message import UserMessage
    from .schema.request import ChatCompletionRequest
    
    async def main():
        req = ChatCompletionRequest(
            messages=[UserMessage(content="设计和艺术有什么区别？")],
            model="deepseek-v4-flash"
        )

        print("AI (Streaming):")
        rflag, tflag = False, False
        async for resp in client.stream_chat(req):
            choice = resp.choices[0]
            delta = choice.delta
            
            if delta.reasoning_content:
                if not rflag:
                    print(f"\n\n[Reasoning]\n", end="")
                    rflag = True
                print(f"{delta.reasoning_content}", end="", flush=True)
            elif delta.content:
                if not tflag:
                    print(f"\n\n[Content]\n", end="")
                    tflag = True
                print(f"{delta.content}", end="", flush=True)
            
            if choice.finish_reason:
                print(f"\n\n[Finish Reason]: {choice.finish_reason}")

    asyncio.run(main())
