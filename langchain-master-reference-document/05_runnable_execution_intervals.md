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

# 5. Runnable Execution Internals

Section 4 showed **how Runnables are composed**.  
This section focuses on what happens when that composed Runnable actually executes.

The key mental model is:

> **A Runnable is not merely a function call; it participates in a managed execution lifecycle.**

---

## 5.1 The Execution Lifecycle

Consider:

```text
A → B → C
```

When you execute:

```python
chain.invoke(input)
```

conceptually:

```text
invoke(input)
    │
    ▼
Start execution
    │
    ▼
Execute A
    │
    ▼
Pass A's output to B
    │
    ▼
Execute B
    │
    ▼
Pass B's output to C
    │
    ▼
Execute C
    │
    ▼
Return final output
```

The important point is that **the composition defines the execution structure**.

---

## 5.2 Nested Runnables

A Runnable can contain other Runnables.

For example:

```text
Application
    │
    ▼
RunnableSequence
    │
    ├── Prompt
    │
    ├── RunnableParallel
    │       ├── Retriever
    │       └── QueryProcessor
    │
    └── Model
```

When the outer Runnable executes, the inner Runnables execute as part of the same execution tree.

This hierarchy becomes important for:

- tracing
- callbacks
- metadata
- error propagation
- configuration propagation
- debugging

---

## 5.3 Input/Output Propagation

For a sequence:

```text
A → B → C
```

the runtime effectively performs:

```text
input_A
   │
   ▼
 A.invoke()
   │
output_A
   │
   ▼
 B.invoke(output_A)
   │
output_B
   │
   ▼
 C.invoke(output_B)
   │
output_C
```

Therefore, composition does **not** eliminate type compatibility requirements.

If:

```text
A : Runnable[str, Message]
B : Runnable[Message, AIMessage]
```

then:

```text
A | B
```

is naturally compatible.

But:

```text
A : Runnable[str, int]
B : Runnable[Message, AIMessage]
```

is conceptually invalid because the data contract doesn't match.

Python's runtime will not necessarily catch every such mismatch before execution, so schema/type discipline remains important.

---

## 5.4 Synchronous Execution

`invoke()` represents a synchronous execution boundary.

Conceptually:

```text
Caller
  │
  │ invoke()
  ▼
Runnable
  │
  ├── execute
  ├── wait
  └── return
  │
  ▼
Result
```

If a pipeline contains three network calls:

```text
A → B → C
```

the caller waits until the complete pipeline finishes.

This is appropriate when:

- the surrounding application is synchronous
- the operation is short-lived
- streaming isn't required

But it becomes problematic when the application itself is asynchronous.

---

## 5.5 Asynchronous Execution

`ainvoke()` provides the asynchronous equivalent.

Conceptually:

```text
await chain.ainvoke(input)
```

The important distinction is not that LangChain somehow makes every operation magically asynchronous.

The underlying implementation still determines whether an operation can perform genuine non-blocking I/O.

For example:

```text
async Runnable
     │
     ▼
non-blocking provider call
```

is genuinely asynchronous.

But wrapping a blocking operation inside an async function doesn't automatically make the underlying operation non-blocking.

This distinction matters heavily in high-concurrency services.

---

## 5.6 Batch Execution

Suppose you have:

```text
input_1
input_2
input_3
...
input_N
```

Instead of:

```python
for item in inputs:
    chain.invoke(item)
```

you can use:

```text
chain.batch(inputs)
```

The Runnable abstraction can then coordinate execution across multiple inputs.

Conceptually:

```text
             ┌──► input 1 ──► execution
             │
batch ───────┼──► input 2 ──► execution
             │
             ├──► input 3 ──► execution
             │
             └──► input N ──► execution
```

The exact concurrency behavior depends on the Runnable implementation and configuration.

---

## 5.7 Why Batching Matters

For LLM applications, batching can improve throughput.

Without batching:

```text
Request 1 → Model
Request 2 → Model
Request 3 → Model
```

With appropriate batching/concurrency:

```text
             ┌── Request 1
             ├── Request 2
Scheduler ───┼── Request 3
             └── Request 4
```

But there are two different concepts that should not be confused:

### Application-level concurrency

Multiple independent requests execute concurrently.

### Provider-native batching

The provider itself receives multiple inputs in a single API operation.

LangChain's `batch()` does not universally imply provider-native batching.

This distinction is important when reasoning about:

- latency
- throughput
- API quotas
- token pricing
- connection usage

---

## 5.8 Streaming Execution

`stream()` changes the execution model.

Instead of:

```text
input
 │
 ▼
complete execution
 │
 ▼
final result
```

you can receive incremental output:

```text
input
 │
 ▼
execution
 │
 ├── chunk 1
 ├── chunk 2
 ├── chunk 3
 ├── chunk 4
 └── ...
       │
       ▼
   final output
```

For an LLM, these chunks commonly represent incremental portions of generated content.

The application can therefore begin processing or displaying output **before the model has finished generating the complete response**.

---

## 5.9 Streaming Is Not Just Faster

Streaming primarily improves **perceived latency**.

Consider:

```text
TTFT = 800 ms
Generation = 4 seconds
```

Without streaming:

```text
User waits ≈ 4.8 seconds
```

With streaming:

```text
User sees first output ≈ 800 ms
```

The total generation time may remain approximately the same.

Therefore:

> **Streaming primarily improves time-to-first-token/user experience, not necessarily total execution time.**

---

## 5.10 Chunk Propagation

One subtle aspect of LangChain streaming is that chunks can propagate through a pipeline.

Conceptually:

```text
Model
  │
  ├── chunk A
  ├── chunk B
  └── chunk C
       │
       ▼
Downstream Runnable
       │
       ▼
Transformed chunks
```

Whether a downstream component can stream effectively depends on whether it supports incremental processing.

A poorly designed transformation can effectively become a streaming bottleneck:

```text
Model ──stream──► Buffer everything ──► Parser
                         │
                         ▼
                   Streaming lost
```

This is a common production concern.

---

## 5.11 Error Propagation

Suppose:

```text
A → B → C
```

and `B` fails.

The normal execution path becomes:

```text
A
│
▼
B ──X
```

`C` does not receive an input.

The error propagates back through the execution hierarchy.

Production systems therefore need to distinguish:

```text
Transient failure
    │
    ├── retry
    │
    └── fallback

Permanent failure
    │
    └── fail request
```

This becomes particularly important for provider APIs where timeouts, rate limits, and temporary availability failures are common.

---

## 5.12 Cancellation

Async execution introduces another important concern:

```text
Client
  │
  │ request
  ▼
Runnable
  │
  ├── Model call
  ├── Retrieval
  └── Tool call
```

If the client disconnects or the request deadline expires, continuing all downstream work can waste:

- model tokens
- compute
- database connections
- network resources

Production systems should therefore consider **cancellation and deadline propagation**, particularly around streaming and long-running workflows.

---

## 5.13 The Execution Tree

The most useful mental model is to think of execution as a tree rather than simply a chain.

```text
Root Runnable
│
├── Prompt
│
├── Parallel
│   ├── Retriever
│   └── Classifier
│
└── Model
    └── Tool
```

Every execution has context associated with it.

That context becomes the foundation for the next section:

> **`RunnableConfig` controls how a Runnable executes at runtime.**

### Key takeaway

`Runnable` composition defines **what executes**;  
The Runnable execution model defines **how that composition is invoked, streamed, batched, traced, configured, and how failures propagate through it**.
