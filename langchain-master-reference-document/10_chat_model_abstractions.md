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

# 10. Chat Model Abstractions

A **chat model abstraction** provides a common interface for interacting with different LLM providers while preserving the message-based execution model we discussed in Section 9.

The key idea is:

> **`BaseChatModel` defines the LangChain-side contract; provider integrations implement the actual communication with the model provider.**

---

## 10.1 The Architecture

Consider an application using different providers:

```text
                  Your Application
                         │
                         ▼
                 BaseChatModel API
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
        OpenAI Model  Anthropic   Google
             │           │           │
             ▼           ▼           ▼
          Provider APIs / SDKs
```

Your application can therefore operate primarily against a common interface rather than embedding provider-specific API calls throughout the application.

---

## 10.2 `BaseChatModel`

At the abstraction level, a chat model is conceptually:

```text
BaseChatModel

Input:
    messages

Output:
    AIMessage
```

So:

```text
List[BaseMessage]
        │
        ▼
   Chat Model
        │
        ▼
    AIMessage
```

This fits naturally into the `Runnable` model discussed earlier.

A chat model is therefore both:

```text
Chat Model
    │
    └── Runnable[LanguageModelInput, AIMessage]
```

This is an important connection between the previous sections.

---

## 10.3 Why a Chat Model Is a Runnable

Because a chat model implements the Runnable protocol, you can compose it:

```text
Prompt
   │
   ▼
Chat Model
   │
   ▼
Parser
```

and execute the complete pipeline with:

```python id="d9efqx"
chain.invoke(...)
```

The model doesn't require a completely different execution mechanism.

It participates in the same:

- `invoke`
- `ainvoke`
- `batch`
- `abatch`
- `stream`
- `astream`
- configuration
- callback

ecosystem.

---

## 10.4 Provider Integration

A provider-specific class is responsible for translating LangChain's normalized representation into the provider's native API.

Conceptually:

```text
LangChain
   │
   │ messages
   ▼
Provider Integration
   │
   │ provider-specific request
   ▼
Provider API
   │
   ▼
Model
   │
   │ provider-specific response
   ▼
Provider Integration
   │
   ▼
AIMessage
```

This translation boundary is important.

Your application doesn't need to understand every provider's HTTP schema.

---

## 10.5 Model Construction vs Model Invocation

There are two separate concerns:

### Construction

```text
Which model?
Which provider?
Which configuration?
```

### Invocation

```text
What messages should be sent?
How should the response be consumed?
```

Conceptually:

```text
Model configuration
      │
      ▼
ChatModel instance
      │
      ▼
messages
      │
      ▼
AIMessage
```

Keeping these concerns separate makes model selection and runtime configuration easier.

---

## 10.6 The Model Does Not Own Your Conversation

This is an important architectural distinction.

Suppose you have:

```text
User A
   │
   ▼
Conversation history
   │
   ▼
Chat Model
```

The chat model generally receives the messages you provide.

The model abstraction does **not automatically constitute your application's durable conversation store**.

Your application or orchestration layer is responsible for deciding:

- what history to retain
- what messages to send
- how much context to include
- when to summarize
- where state is persisted

This becomes especially important when we later discuss LangGraph state and memory.

---

## 10.7 Model Configuration

A chat model usually exposes configuration such as:

```text
model
temperature
max output tokens
stop sequences
provider-specific parameters
```

But there is an architectural distinction between:

```text
Common model configuration
```

and:

```text
Provider-specific configuration
```

For example:

```text
                    ChatModel
                       │
          ┌────────────┴────────────┐
          │                         │
   common parameters        provider-specific
                                  parameters
```

The abstraction can normalize common capabilities, but it cannot guarantee that every provider supports every parameter identically.

---

## 10.8 Capability Differences

Suppose provider A supports:

```text
tool calling
structured output
multimodal input
reasoning configuration
```

while provider B supports only:

```text
tool calling
text input
```

A common abstraction cannot magically make provider B support the missing capabilities.

Therefore:

> **An abstraction standardizes the interface, not the underlying model capabilities.**

This is one of the most important limitations to understand when designing provider-independent systems.

---

## 10.9 Streaming

Chat models also participate in LangChain's streaming model.

Instead of:

```text
messages
   │
   ▼
complete AIMessage
```

streaming produces incremental chunks:

```text
messages
   │
   ▼
Chat Model
   │
   ├── AIMessageChunk
   ├── AIMessageChunk
   ├── AIMessageChunk
   └── ...
```

These chunks can then propagate through a Runnable pipeline.

For example:

```text
Prompt
  │
  ▼
Model
  │
  ├── chunk 1
  ├── chunk 2
  └── chunk 3
       │
       ▼
Streaming consumer
```

The provider integration is responsible for translating the provider's native streaming protocol into LangChain's streaming representation.

---

## 10.10 Usage Metadata

Modern chat model responses can expose usage information.

Conceptually:

```text
AIMessage
├── content
├── response_metadata
└── usage_metadata
       ├── input tokens
       ├── output tokens
       └── total tokens
```

This allows your application to calculate:

```text
cost / request
tokens / request
tokens / workflow step
```

However, provider metadata is not necessarily identical across providers.

For production systems, treat usage metadata as an **integration contract that must be validated for the providers you actually support**, rather than assuming perfect uniformity.

---

## 10.11 Native SDK vs LangChain Model

The architectural trade-off can be summarized as:

| Native Provider SDK | LangChain Chat Model |
|---|---|
| Maximum provider-specific control | Common interface |
| Direct access to provider features | Easier composition |
| Smaller abstraction surface | Runnable ecosystem |
| Provider-specific implementation | Easier provider substitution |
| Less framework coupling | Better integration with LangChain tooling |

For a simple application:

```text
Application → Provider SDK
```

may be preferable.

For a larger system:

```text
Application
    │
    ▼
LangChain Runnable pipeline
    │
    ▼
Chat Model
    │
    ▼
Provider
```

the abstraction can significantly simplify composition.

---

## 10.12 The Important Boundary

A good production architecture does **not** assume:

```text
LangChain = model
```

Instead:

```text
                LangChain
                    │
             abstraction layer
                    │
                    ▼
             Provider API
                    │
                    ▼
             Foundation Model
```

LangChain controls how your application interacts with the model.

The provider controls:

- model implementation
- inference infrastructure
- model-specific capabilities
- API semantics
- provider-side limits

---

## Quick Recall

| Concept | Meaning |
|---|---|
| `BaseChatModel` | Common chat-model abstraction |
| Input | Messages / language-model input |
| Output | `AIMessage` or streamed chunks |
| Provider integration | Translates LangChain ↔ provider API |
| Runnable | Enables standard LangChain execution |
| Streaming | Incremental `AIMessageChunk` output |
| Usage metadata | Token/response accounting |
| Capability boundary | Provider-specific features still exist |

### Key takeaway

> **A LangChain chat model is a Runnable abstraction over a provider's conversational model API. It normalizes invocation, messages, streaming, configuration, and metadata while deliberately leaving actual model inference to the provider.**

---
