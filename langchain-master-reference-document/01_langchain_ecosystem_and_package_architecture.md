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

# 1. LangChain Ecosystem & Package Architecture

LangChain is best understood as a **set of composable abstractions for building LLM applications**, rather than a single monolithic framework.

## 1.1 The High-Level Architecture

The modern ecosystem can be viewed roughly as:

```text
                   Your Application
                          │
                          ▼
                 ┌─────────────────┐
                 │    LangGraph    │
                 │  Orchestration  │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │    LangChain    │
                 │  High-level AI  │
                 │  application    │
                 │  components     │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ langchain-core  │
                 │ Runnable /      │
                 │ Messages /      │
                 │ Tools / Schemas │
                 └────────┬────────┘
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
         OpenAI       Anthropic      Google
         Provider     Provider       Provider
```

The important architectural idea is:

> **`langchain-core` defines the contracts; integration packages implement those contracts; LangChain provides higher-level building blocks; LangGraph provides explicit orchestration.**

---

## 1.2 `langchain-core`

`langchain-core` contains the fundamental abstractions used throughout the ecosystem.

Examples:

- `Runnable`
- `RunnableLambda`
- `RunnableParallel`
- `RunnablePassthrough`
- `BaseMessage`
- `AIMessage`
- `HumanMessage`
- `BaseChatModel`
- `BaseRetriever`
- `BaseTool`
- Prompt templates
- Output parsers
- Callback interfaces
- `RunnableConfig`

Think of it as the **runtime contract layer**.

For example:

```text
Runnable[A, B]
A ──invoke()──► B
```

Once something implements the `Runnable` contract, it can participate in the same composition mechanism regardless of whether it represents:

- a prompt
- an LLM
- a retriever
- a Python function
- a parser
- a complete sub-pipeline

This is the foundation of LCEL.

---

## 1.3 Provider Integration Packages

Provider-specific implementations live separately.

Conceptually:

```text
langchain-openai
langchain-anthropic
langchain-google-genai
...
        │
        ▼
langchain-core interfaces
```

For example, an OpenAI chat model implements LangChain's chat-model abstractions while internally communicating with OpenAI's API.

This separation is important because **LangChain does not implement the model itself**.

The actual execution remains:

```text
Your application
      │
      ▼
LangChain abstraction
      │
      ▼
Provider SDK / API
      │
      ▼
Foundation model
```

Therefore, LangChain is primarily an **orchestration and abstraction layer**, not an inference engine.

---

## 1.4 `langchain`

The `langchain` package provides higher-level components built on top of `langchain-core`.

Examples include:

- Retrieval utilities
- Agent building blocks
- Document processing
- Prompt utilities
- Higher-level integrations between components

The distinction is useful:

```text
langchain-core
    ↓
primitive contracts

langchain
    ↓
higher-level application components
```

You should generally understand `core` before relying heavily on higher-level abstractions.

---

## 1.5 `langchain-community`

Historically, LangChain accumulated a very large number of integrations.

Many of these integrations were moved into `langchain-community`.

Examples include integrations around:

- document loaders
- vector stores
- third-party services
- miscellaneous tooling

Architecturally, this prevents the core packages from becoming tightly coupled to hundreds of external dependencies.

---

## 1.6 LangGraph

LangGraph solves a different problem.

LangChain gives you composable execution:

```text
A → B → C → D
```

LangGraph gives you explicit stateful orchestration:

```text
       ┌───────┐
       │ Node A│
       └───┬───┘
           │
           ▼
       ┌───────┐
       │ Node B│
       └───┬───┘
           │
       ┌───▼───┐
       │ Node C│
       └───┬───┘
           │
       ┌───▼───┐
       │ Node D│
       └───────┘
```

But unlike a simple chain, the graph can contain:

- cycles
- conditional routing
- persistent state
- interrupts
- human approval
- checkpoints
- resumable execution

So the relationship is:

```text
                 LangGraph
                     │
                orchestration
                     │
                     ▼
                LangChain Core
                     │
              execution primitives
                     │
                     ▼
             Provider integrations
```

LangGraph does **not replace** LangChain's primitives. It can use them.

---

## 1.7 What You Actually Need to Remember

| Package                   | Primary responsibility                           |
| ------------------------- | ------------------------------------------------ |
| `langchain-core`          | Fundamental interfaces and execution primitives  |
| `langchain`               | Higher-level application components              |
| `langchain-community`     | Community-maintained integrations                |
| `langchain-openai` / etc. | Provider-specific implementations                |
| `langgraph`               | Stateful, graph-based orchestration              |
| LangSmith                 | Observability, tracing, evaluation and debugging |

### Mental model

```text
                  APPLICATION
                      │
                      ▼
                  LANGGRAPH
              state + control flow
                      │
                      ▼
                  LANGCHAIN
            application abstractions
                      │
                      ▼
                LANGCHAIN-CORE
          Runnable + messages + tools
                      │
                      ▼
            PROVIDER INTEGRATIONS
                      │
                      ▼
                 MODEL / API
```

The **most important takeaway for the rest of this document** is that `langchain-core` is the foundation.  
Once we understand its `Runnable` protocol, much of the rest of LangChain becomes composition of those same primitives.

---

[Docs by LangChain](https://docs.langchain.com/oss/python/langchain/overview)

---
