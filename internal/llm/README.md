- [x] Background

- [x] Entity
- [~] Adaptor (只完成deepseek的适配)
- [~] Streaming Protocol (只完成deepseek的适配)
- [] Err
- [] Cancel
- [] Schema

- [] Plugin Manager

# llm

围绕 LLM 的软件设计，其解决的问题围绕着这个自然语言的不确定性计算工具展开的。

此组件 (internal.llm) 将描述 Agent Harness 所期待的 llm 基础设施应当具备的能力。

# 1 Entity

## 1.1 Message & Block

与 Agent 的交流消息模型，至少需要解决两个问题：

1. 消息是什么？
2. 以什么形式去组织消息？

### Message & Role

最初的广为人知的 AI 应用莫过于ChatGPT，这是人们对于 AI 应当拥有能力的最初认知。

参考 LangChain 中的消息建模，最简单的消息模型是：

- HumanMessage: 即人类发送的消息
- AIMessage: 即 AI 所发送的消息

这种层次的形成是 ChatModel 顺而形成的一个惯性认识————人类提出一个问题，Agent根据人类的要求去解决问题。

### Message's Content: Blocks

现代人机交互模型中，Message 的内容并非是纯粹的 string 形式，我们会讨论最近产生的新技术，以及它们对于 Message 产生的影响。

#### i. Reasoning

最初，研究人员提出了许多早期的 Reasoning 范式，例如 CoT, ToT 等等。人们发现，引导模型在直接回答之前先"想一想"，能够显著提高模型的回答质量。

几乎任何回答都可以因此受益，其作用的广泛性使得模型的提供方很快将其划入了模型训练中的重要步骤。

目前，人们最广泛使用的方式是先让模型输出以`<think></think>`包裹的思考过程，然后再输出回答。

这使得 **Reasoning Block** 在 Message 中自然而然地出现。

#### ii. Tool Calling

随着 llm 的发展，人们开始认识到：在纯自然语言领域进行的运算存在着局限性，例如：

- AI 模型在只使用纯自然语言手段推理的情况下，对高精度数值计算几乎束手无策；
- 由于训练十分昂贵，AI 模型无法通过纯训练的方式频繁更新其具备的知识，这也是 RAG 的问题背景；

研究者们为 LLM 扩展了能力，给予了它使用外部工具的能力，并设计了 MCP 以规范模型调用工具的方式。

例如：

- 为了让模型能够执行高精度计算，我们让 LLM 能够调用计算器；
- 为了让模型能够快速获取最新的知识，我们给予它 WebSearch 或者 RAG 的能力。

这说明了近些年来大语言模型能力扩展的最重要的方向，也就是 Tool Calling。它要求Message中具备：

- 模型发起 Tool Calling 的消息模型
- Tool Calling 执行并返回其结果的消息模型

这也就是 **Tool-call Block** 和 **Tool-result Block** 的来源。

上述的三个 Blocks，加上模型的返回给用户的最终回答 **Text Blocks**，成为了 AI Agent的四个核心消息模型底座。


# 2 Stream Protocol

