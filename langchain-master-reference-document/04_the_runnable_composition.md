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

# 4. Runnable Composition

`Runnable` becomes particularly powerful when multiple Runnables are composed into a single execution pipeline.

The central idea is:

> **A composed Runnable is itself a Runnable.**

This gives LangChain a compositional execution model rather than a collection of unrelated utilities.

---

## 4.1 Sequential Composition — `|`

The most fundamental composition mechanism is the pipe operator:

```text
A → B → C
```

For example:

```python
chain = prompt | model | parser
```

Conceptually:

```text
Input
  │
  ▼
Prompt
  │
  ▼
Chat Model
  │
  ▼
Parser
  │
  ▼
Output
```

The output type of one component must be compatible with the input type expected by the next component.

So:

```text
Runnable[A, B] | Runnable[B, C]
                    │
                    ▼
              Runnable[A, C]
```

This is essentially **function composition with LangChain execution semantics**.

---

## 4.2 `RunnableSequence`

The pipe syntax is syntactic sugar for creating a sequential composition.

Conceptually:

```python
chain = a | b | c
```

produces something equivalent to:

```text
RunnableSequence
    ├── a
    ├── b
    └── c
```

The important consequence is that LangChain now sees the entire sequence as **one Runnable**.

Therefore you can do:

```text
chain.invoke(...)
chain.ainvoke(...)
chain.batch(...)
chain.abatch(...)
chain.stream(...)
chain.astream(...)
```

without individually managing every component.

---

## 4.3 `RunnableLambda`

A normal Python function can be adapted into the Runnable model.

Conceptually:

```text
Python function
      │
      ▼
RunnableLambda
      │
      ▼
   Runnable
```

This is important because it allows application-specific deterministic logic to participate in an LCEL pipeline.

For example:

```text
Retriever
    │
    ▼
Python transformation
    │
    ▼
  Prompt
    │
    ▼
  Model
```

The transformation does not need to be a special LangChain component.

---

## 4.4 `RunnablePassthrough`

Sometimes you don't want to transform the input.

You want to **carry it forward**.

That's the role of `RunnablePassthrough`.

Conceptually:

```text
input
  │
  ├──────────────► downstream A
  │
  └──Passthrough─► downstream B
```

This becomes particularly useful when constructing dictionaries of inputs.

For example, conceptually:

```text
{
    "context": retriever,
    "question": RunnablePassthrough()
}
```

produces:

```text
question
   │
   ├──► retriever ──► context
   │
   └──► passthrough ──► question
```

Result:

```python
{
    "context": [...],
    "question": "original question"
}
```

This pattern is extremely common in RAG pipelines.

---

## 4.5 `RunnableParallel`

`RunnableParallel` allows multiple Runnables to execute against the same input.

Conceptually:

```text
                    ┌──► Runnable A
                    │
Input ──────────────┼──► Runnable B
                    │
                    └──► Runnable C
```

The result is collected into a dictionary:

```text
{
    "a": result_a,
    "b": result_b,
    "c": result_c
}
```

This is useful when operations are independent.

For example:

```text
                  ┌──► Retriever
Question ─────────┤
                  └──► Query Analyzer
```

There is no reason for the analyzer to wait for the retriever if both only depend on the original question.

---

## 4.6 Sequential vs Parallel Composition

This distinction matters considerably for production systems.

### Sequential

```text
A ──► B ──► C
```

Latency is approximately:

```text
T ≈ T_A + T_B + T_C
```

### Parallel

```text
       ┌──► A ──┐
Input ─┤        ├──► Result
       └──► B ──┘
```

Latency is approximately:

```text
T ≈ max(T_A, T_B)
```

assuming sufficient concurrency and no downstream bottleneck.

This can produce substantial latency improvements.

But it also increases:

- concurrent provider requests
- rate-limit pressure
- memory consumption
- connection usage
- cost if every branch performs model calls

So parallelism is **not automatically better**.

---

## 4.7 `RunnableBranch`

`RunnableBranch` introduces conditional execution.

Conceptually:

```text
                 ┌── condition A ──► A
Input ───────────┤
                 ├── condition B ──► B
                 │
                 └── otherwise ────► C
```

This is useful when routing should be deterministic.

For example:

```text
Query
 │
 ├── technical? ──► Technical pipeline
 │
 ├── billing? ────► Billing pipeline
 │
 └── otherwise ──► General pipeline
```

This is fundamentally different from asking an LLM to decide which pipeline to execute.

If the routing rule can be expressed deterministically, deterministic routing is generally preferable.

---

## 4.8 Composition Creates a Graph

Even a simple:

```python
a | b | c
```

can be thought of as:

```text
A
│
▼
B
│
▼
C
```

While:

```text
{
    "x": a,
    "y": b
}
```

creates:

```text
       ┌──► A ──► x
Input ─┤
       └──► B ──► y
```

This is why LangChain pipelines naturally lend themselves to:

- visualization
- tracing
- callbacks
- execution metadata
- debugging

The composition itself carries structural information.

---

## 4.9 The Core Composition Primitives

| Primitive | Purpose |
|---|---|
| `RunnableSequence` / `\|` | Sequential execution |
| `RunnableParallel` | Concurrent independent execution |
| `RunnableLambda` | Adapt Python logic |
| `RunnablePassthrough` | Preserve/pass input |
| `RunnableBranch` | Conditional routing |
| `RunnableAssign` | Add computed values to an existing dictionary |

These primitives are enough to express surprisingly complex **deterministic pipelines**.

---

## 4.10 The Architectural Boundary

There is an important limit.

LCEL composition works particularly well for:

```text
A → B → C
```

and:

```text
       ┌── B
A ─────┤
       └── C
```

But once the system requires:

```text
A
│
▼
B
│
▼
C ──────┐
▲       │
│       ▼
└──── D ◄── human approval
        │
        ▼
      persist
        │
        ▼
      resume
```

you are moving beyond simple pipeline composition.

That is where **LangGraph's explicit stateful orchestration model** becomes relevant.

### Key takeaway

> **LCEL composition gives you a functional execution graph: sequence, parallelism, transformation, passthrough, and deterministic routing.**

The important engineering question is not *"Can I express this with LCEL?"* but **"Does this remain a deterministic dataflow pipeline, or have I crossed into stateful orchestration?"**
