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

# 2. LangChain Design Philosophy

The central idea behind modern LangChain is:

> **Provide composable abstractions around LLM application primitives without owning the application's business logic.**

## 2.1 Composition Over Monolithic Chains

Modern LangChain favors small components that implement common interfaces.

```text
Prompt
   │
   ▼
Chat Model
   │
   ▼
Parser
```

Each component can be independently replaced, tested, configured, streamed, or composed.  

This is fundamentally different from older abstractions such as `LLMChain`, where multiple concerns were bundled into one object.  

---

## 2.2 Standard Interfaces

LangChain attempts to give different operations common interfaces.

The most important one is:

```text
Runnable[A, B]
```

For example:

```text
PromptTemplate
      │
      ▼
ChatModel
      │
      ▼
OutputParser
```

All can participate in the same execution model.

This provides **polymorphism at the application layer**.

You can replace:

```text
OpenAI
```

with:

```text
Anthropic
```

without necessarily redesigning the surrounding pipeline.

---

## 2.3 Separate Deterministic Logic from Probabilistic Logic

A good LangChain application does not give the LLM control over everything.

Instead:

```text
Deterministic application logic
          │
          ├── validation
          ├── authorization
          ├── persistence
          ├── retries
          └── routing
                  │
                  ▼
          Probabilistic model
                  │
                  ▼
          Structured result
                  │
                  ▼
          Deterministic logic
```

The model should generally make **bounded decisions**, while your application owns critical invariants.

This becomes particularly important when building agents.

---

## 2.4 Framework Abstractions Are Optional

`LangChain` is not intended to replace ordinary Python.

For a simple operation:

```python
response = await openai_client.responses.create(...)
```

adding LangChain may provide little value.

For a system involving:

- multiple model providers
- streaming
- tools
- retrieval
- structured outputs
- callbacks
- retries
- graph orchestration

the abstractions become considerably more useful.

So the correct question is not:

> "Can LangChain do this?"

It is:

> **"Does LangChain reduce enough application complexity to justify its abstraction and dependency cost?"**

---

## 2.5 Abstraction Has a Cost

LangChain introduces another layer between your application and the provider.

```text
Application
    │
    ▼
LangChain
    │
    ▼
Provider SDK
    │
    ▼
Provider API
```

That layer provides consistency, but introduces potential costs:

- additional dependency surface
- framework version churn
- abstraction leakage
- debugging complexity
- provider features that may not map cleanly to common interfaces

Therefore, **provider-native APIs remain important even when using LangChain**.

---

## 2.6 The Core Engineering Principle

For production systems, think of LangChain as:

```text
              LangChain
                  │
       ┌──────────┴──────────┐
       │                     │
   Abstraction           Composition
       │                     │
       └──────────┬──────────┘
                  ▼
          Application Logic
```

It should help you **compose and operate AI components**, not dictate your entire system architecture.

### Key takeaway

**LangChain is most valuable when the complexity of composing and operating LLM components exceeds the complexity introduced by the framework itself.**

This principle will be important later when we compare **plain Python vs. LangChain vs. LangGraph**.
