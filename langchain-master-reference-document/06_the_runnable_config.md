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

# 6. `RunnableConfig`

`RunnableConfig` is the **runtime control plane for a Runnable execution**.

While the Runnable defines *what* should execute, `RunnableConfig` lets the caller influence *how that execution should behave*.

---

## 6.1 Why `RunnableConfig` Exists

Consider:

```text
Application
    │
    ▼
Runnable
    │
    ├── execution
    ├── callbacks
    ├── tracing
    ├── metadata
    └── concurrency
```

You don't want these concerns hard-coded into every Runnable.

Instead, they are supplied at runtime:

```python
result = chain.invoke(
    input,
    config=config,
)
```

This gives us a clean separation:

```text
Runnable definition
        │
        │ what to execute
        ▼
Runnable
        ▲
        │ how to execute
        │
RunnableConfig
```

---

# 6.2 What `RunnableConfig` Controls

The configuration can carry several categories of runtime information.

The important ones are:

| Configuration | Purpose |
|---|---|
| `tags` | Categorize and identify runs |
| `metadata` | Attach structured contextual information |
| `callbacks` | Observe execution lifecycle |
| `max_concurrency` | Limit concurrent execution |
| `recursion_limit` | Prevent uncontrolled recursive execution |
| `configurable` | Supply runtime-configurable values |

Not every Runnable uses every configuration field.

---

# 6.3 Basic Example

A simple example:

```python
from langchain_core.runnables import RunnableLambda

def uppercase(value: str) -> str:
    return value.upper()

runnable = RunnableLambda(uppercase)

result = runnable.invoke(
    "hello",
    config={
        "tags": ["example"],
        "metadata": {
            "request_id": "req-123",
        },
    },
)

print(result)
```

The important point is that the business function remains:

```python
def uppercase(value: str) -> str:
    return value.upper()
```

It doesn't need to know anything about:

- request IDs
- tracing
- LangChain callbacks
- execution metadata

That information belongs to the **execution context**, not the business function.

---

# 6.4 Configuration Propagation

This is one of the most important concepts.

Suppose:

```text
Root Runnable
     │
     ▼
Runnable A
     │
     ▼
Runnable B
```

and the caller provides:

```text
metadata = {
    "request_id": "abc"
}
```

Conceptually:

```text
                    request_id=abc
                           │
                           ▼
                       Root
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
                 A                   B
                 │                   │
                 └──── propagated ───┘
```

Nested Runnables can therefore participate in the same execution context.

This is extremely useful for distributed tracing.

---

# 6.5 Tags vs Metadata

These are easy to confuse.

### Tags

Tags are primarily **categorical labels**.

```python
tags=["production", "rag"]
```

Think:

```text
"What kind of run is this?"
```

### Metadata

Metadata contains **structured contextual information**.

```python
metadata={
    "request_id": "req-123",
    "tenant_id": "tenant-42",
}
```

Think:

```text
"What contextual information belongs to this run?"
```

A useful rule:

```text
Tags     → classification
Metadata → context
```

---

# 6.6 `max_concurrency`

This controls how much parallel work a Runnable is allowed to execute concurrently.

For example:

```python
config = {
    "max_concurrency": 5,
}
```

Conceptually:

```text
100 inputs
    │
    ▼
┌─────────────────┐
│ concurrency = 5 │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
  task 1    task 2
  task 3    task 4
  task 5
```

The remaining tasks wait until execution capacity becomes available.

### Why this matters

Without concurrency control:

```text
Application
    │
    ├──► Model
    ├──► Model
    ├──► Model
    ├──► Model
    ├──► ...
    └──► Model
```

you can easily overwhelm:

- provider rate limits
- connection pools
- CPU
- memory
- downstream services

Therefore `max_concurrency` is an **operational control**, not merely a convenience.

---

# 6.7 Concurrency Is Not Throughput

A common mistake is:

> "Higher concurrency means higher throughput."

Only up to a point.

Suppose a provider permits:

```text
100 requests/sec
```

but your application sends:

```text
500 concurrent requests
```

You may simply produce:

```text
rate limits
    ↓
retries
    ↓
more requests
    ↓
more rate limits
```

which can create a feedback loop.

A production system should therefore consider:

```text
Application concurrency
        ↓
Provider limits
        ↓
Retry policy
        ↓
Backpressure
```

`RunnableConfig` gives you one important control point within that system.

---

# 6.8 Recursion Limits

Some Runnable compositions can recursively invoke other Runnables.

Without a termination boundary, recursive execution can become pathological.

`recursion_limit` provides a guardrail against uncontrolled recursion.

Conceptually:

```text
A
│
▼
B
│
▼
A
│
▼
B
│
▼
...
```

A recursion limit prevents this from continuing indefinitely.

This is particularly relevant when execution logic dynamically constructs or invokes additional Runnable operations.

---

# 6.9 Runtime-Configurable Values

One of the more powerful capabilities is separating **pipeline definition** from **runtime configuration**.

Conceptually:

```text
Pipeline definition
       │
       ▼
 ┌───────────────┐
 │ Model         │
 │ configurable  │
 └───────┬───────┘
         ▲
         │
         │ runtime config
         │
 ┌───────┴───────┐
 │ provider/model│
 └───────────────┘
```

This allows an application to construct a pipeline once while selecting certain parameters at runtime.

For example:

```text
Request A → model = provider-A
Request B → model = provider-B
```

without duplicating the entire pipeline.

This becomes useful for:

- A/B testing
- model routing
- tenant-specific configuration
- fallback architectures
- controlled experimentation

---

# 6.10 Configuration Is Not Application State

This distinction is important.

Avoid treating `RunnableConfig` as a general-purpose state store.

For example, don't conceptually use it to hold:

```text
conversation history
shopping cart
workflow state
database entities
agent state
```

Those belong to application/workflow state.

A useful separation is:

```text
RunnableConfig
    │
    └── execution context

Application State
    │
    └── business/workflow data
```

This distinction becomes particularly important once we reach LangGraph.

---

# 6.11 Why This Matters for Production

Imagine an HTTP request:

```text
POST /chat
request_id = abc123
tenant = customer-42
```

You execute:

```text
HTTP request
    │
    ▼
LangChain pipeline
    │
    ├── Retriever
    │
    ├── Prompt
    │
    ├── Model
    │
    └── Parser
```

You want all of those operations associated with:

```text
request_id = abc123
tenant = customer-42
```

rather than manually passing those values through every function.

That is exactly the kind of concern runtime configuration and callback/context propagation are designed to support.

---

# 6.12 Production Mental Model

Think of `RunnableConfig` as:

```text
                    Runnable
                       │
            ┌──────────┴──────────┐
            │                     │
        Definition            Execution
            │                     │
        "What?"              "How?"
                                  │
                                  ▼
                           RunnableConfig
                                  │
                  ┌───────────────┼───────────────┐
                  │               │               │
                tracing       concurrency     runtime config
```

The Runnable should describe the **computation**.

The configuration describes the **execution context**.

---

## Quick Recall

| Concept | Meaning |
|---|---|
| `Runnable` | Defines the executable operation |
| `RunnableConfig` | Controls runtime execution |
| `tags` | Categorical run labels |
| `metadata` | Structured execution context |
| `callbacks` | Execution lifecycle observers |
| `max_concurrency` | Concurrency/backpressure control |
| `recursion_limit` | Recursive execution guard |
| `configurable` | Runtime-selectable parameters |

### Key takeaway

> **`RunnableConfig` is the execution-context mechanism that lets the same Runnable behave differently across requests without embedding operational concerns into the Runnable itself.**

---
