<!-- markdownlint-disable MD001 -->
<!-- markdownlint-disable MD012 -->
<!-- markdownlint-disable MD022 -->
<!-- markdownlint-disable MD024 -->
<!-- markdownlint-disable MD025 -->
<!-- markdownlint-disable MD026 -->
<!-- markdownlint-disable MD029 -->
<!-- markdownlint-disable MD040 -->
<!-- markdownlint-disable MD056 -->
<!-- markdownlint-disable MD060 -->

# 9. Messages & Message Lifecycle

Messages are the **fundamental data representation for communication with chat models** in modern LangChain.

The important mental model is:

> **A chat model does not fundamentally operate on strings; it operates on structured messages with roles, content, and metadata.**

---

## 9.1 Why Messages Exist

A simple LLM interface might look like:

```text
string → model → string
```

Chat models need more structure:

```text
messages → model → AI message
```

For example:

```text
┌──────────────────────┐
│ SystemMessage        │
│ "You are an assistant"│
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ HumanMessage         │
│ "Explain RAG"        │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ AIMessage            │
│ "RAG is..."          │
└──────────────────────┘
```

Each message carries semantic information that a plain string does not.

---

# 9.2 Core Message Types

The primary message types you need to understand are:

| Message | Represents |
|---|---|
| `SystemMessage` | Instructions/context supplied by the application |
| `HumanMessage` | User input |
| `AIMessage` | Model-generated response |
| `ToolMessage` | Result returned from a tool invocation |

There are additional message abstractions and metadata structures, but these form the core model.

---

## 9.3 `BaseMessage`

These message types share a common conceptual base:

```text id="rrp4f5"
                 BaseMessage
                      │
        ┌─────────────┼──────────────┐
        │             │              │
     System        Human           AI
                                    │
                                    ▼
                                  Tool
```

`BaseMessage` provides the common message representation.

A message generally contains concepts such as:

```text id="y5q9uw"
Message
├── content
├── additional_kwargs
├── response_metadata
├── id
└── type
```

Not every field is necessarily populated for every message.

---

# 9.4 Message Content

The simplest message content is a string:

```python id="v3b2mp"
from langchain_core.messages import HumanMessage

message = HumanMessage(
    content="Explain vector databases."
)
```

But modern model APIs increasingly support **structured content blocks**.

Conceptually:

```text id="g4v4a7"
content
   │
   ├── text
   ├── image
   ├── audio
   └── other provider-supported content
```

This is important for multimodal applications.

Therefore, don't mentally reduce:

```text
message.content
```

to:

```text
always a string
```

---

# 9.5 System Messages

A `SystemMessage` represents instructions supplied by the application.

```text id="c8d4eu"
SystemMessage
     │
     ▼
"You are a technical assistant."
```

It is conceptually different from:

```text
HumanMessage
     │
     ▼
"Explain RAG."
```

The distinction matters because provider APIs may assign different semantics to system/developer/user roles.

LangChain's message abstraction provides a normalized representation while the provider integration handles translation into the provider's native format.

---

# 9.6 Human Messages

`HumanMessage` generally represents user-originated input.

```text id="s0b4r6"
HumanMessage
    │
    ▼
"What is LCEL?"
```

But architecturally, don't assume that every `HumanMessage` must literally come from a human.

An application can construct one programmatically.

The important point is its **semantic role**, not its physical origin.

---

# 9.7 AI Messages

`AIMessage` represents the model's response.

It can contain much more than generated text.

For example:

```text id="7b0e5g"
AIMessage
├── content
├── tool_calls
├── response_metadata
├── usage_metadata
└── additional_kwargs
```

This is particularly important for tool calling.

A model may produce:

```text id="9jjxkz"
AIMessage
     │
     ├── text
     │
     └── tool_calls
          │
          ├── tool = "get_weather"
          └── arguments = {...}
```

The application then executes the requested tool.

---

# 9.8 Tool Messages

A `ToolMessage` represents the result of a tool execution being returned to the model.

The lifecycle becomes:

```text id="sm4w7h"
HumanMessage
      │
      ▼
    Model
      │
      ▼
AIMessage
  tool_call
      │
      ▼
    Tool
      │
      ▼
ToolMessage
      │
      ▼
    Model
      │
      ▼
AIMessage
```

This is one of the fundamental mechanisms behind tool-using agents.

Notice that the model doesn't directly execute the tool.

The application/runtime sits between the model's request and the tool's execution.

---

# 9.9 Tool Call Identity

Tool calls need to be correlated with their results.

Conceptually:

```text id="k0l0xv"
AIMessage
    │
    └── tool_call_id = abc123
                 │
                 ▼
            ToolMessage
                 │
                 └── tool_call_id = abc123
```

This allows the runtime/model conversation to associate:

```text
"this tool result"
```

with:

```text
"this specific tool request"
```

This becomes particularly important when multiple tools are called concurrently.

---

# 9.10 Message Lifecycle

A typical tool-using conversation can therefore look like:

```text id="2zv2o8"
1. HumanMessage
       │
       ▼
2. Chat Model
       │
       ▼
3. AIMessage
       │
       └── tool_calls
               │
               ▼
4. Tool execution
               │
               ▼
5. ToolMessage
               │
               ▼
6. Chat Model
               │
               ▼
7. AIMessage
```

This is the fundamental message lifecycle underlying many modern agent architectures.

---

# 9.11 Messages vs Strings

This distinction becomes increasingly important as systems become more sophisticated.

### String-based pipeline

```text id="w3c2c7"
"question"
    ↓
"answer"
```

### Message-based pipeline

```text id="w8j0pj"
SystemMessage
HumanMessage
ToolMessage
      ↓
   Chat Model
      ↓
AIMessage
```

The message representation preserves semantics that would otherwise need to be encoded manually.

---

# 9.12 Metadata

Messages can also carry metadata.

For example, an `AIMessage` may contain model/provider information such as:

```text id="f0f6oc"
response_metadata
    │
    ├── model information
    ├── provider-specific details
    └── finish information

usage_metadata
    │
    ├── input tokens
    ├── output tokens
    └── total tokens
```

The exact fields vary by provider.

This is extremely useful for production observability and cost accounting.

---

# 9.13 Provider Translation

LangChain's message model provides a normalized representation, but providers have their own wire formats.

Conceptually:

```text id="9gl3o6"
LangChain Messages
       │
       ▼
Provider Integration
       │
       ▼
Provider-specific request format
       │
       ▼
Model
       │
       ▼
Provider response
       │
       ▼
LangChain AIMessage
```

This is one of the practical benefits of the abstraction.

Your application can reason in terms of:

```text
HumanMessage
AIMessage
ToolMessage
```

rather than provider-specific JSON structures everywhere.

---

# 9.14 The Important Trade-off

Normalization is useful, but it cannot completely erase provider differences.

Different providers may support different:

- content types
- tool-calling semantics
- reasoning metadata
- caching mechanisms
- token accounting
- response metadata
- streaming behavior

Therefore:

> **LangChain's message abstraction provides a common semantic model, not perfect feature equivalence between providers.**

This is an important theme throughout LangChain.

---

# Quick Recall

| Message | Role | Typical lifecycle position |
|---|---|---|
| `SystemMessage` | Application instructions | Before model invocation |
| `HumanMessage` | User/application input | Before model invocation |
| `AIMessage` | Model output | After model invocation |
| `ToolMessage` | Tool execution result | Between model invocations |
| `BaseMessage` | Common message abstraction | Foundation |

### Core lifecycle

```text
HumanMessage
      ↓
   AIMessage
      ↓
   Tool call
      ↓
 ToolMessage
      ↓
   AIMessage
```

### Key takeaway

> **Messages are the typed conversational protocol between your application, models, and tools. `AIMessage` is particularly important because it can represent not only generated content but also structured model decisions such as tool calls and usage metadata.**

---
