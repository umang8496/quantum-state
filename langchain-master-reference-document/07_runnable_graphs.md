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

# 7. Runnable Graphs

A **Runnable graph** is the structural representation of a composed Runnable pipeline.

The key idea is:

> **When you compose Runnables, LangChain can represent the resulting execution structure as a graph rather than treating it as an opaque function.**

---

## 7.1 From Pipeline to Graph

Consider:

```python
chain = prompt | model | parser
```

Conceptually, this becomes:

```text
┌────────┐
│ Prompt │
└───┬────┘
    │
    ▼
┌───────┐
│ Model │
└───┬───┘
    │
    ▼
┌────────┐
│ Parser │
└────────┘
```

Each component becomes a node, and the data flow between them becomes an edge.

---

## 7.2 Why Represent It as a Graph?

If LangChain only treated:

```python
prompt | model | parser
```

as a callable, it would know how to execute it but would have limited structural visibility.

A graph representation gives the framework information about:

- what components exist
- how they are connected
- where parallel branches occur
- where execution converges
- how nested Runnables are structured

That information is valuable for **debugging, visualization, tracing, and reasoning about execution**.

---

## 7.3 Sequential Composition

A sequence:

```text
A → B → C
```

is a simple directed acyclic graph:

```text
A
│
▼
B
│
▼
C
```

The direction represents data flow.

If `A` produces:

```text
output_A
```

that becomes the input to `B`.

---

## 7.4 Parallel Composition

Now consider:

```python
{
    "documents": retriever,
    "question": RunnablePassthrough(),
}
```

The structure becomes:

```text
                 ┌─────────────┐
                 │  Retriever  │
                 └──────┬──────┘
                        │
Input ──────────────────┼──────► documents
                        │
                 ┌──────▼──────┐
                 │ Passthrough │
                 └──────┬──────┘
                        │
                        ▼
                    question
```

The important property is that both branches depend on the same upstream input.

This gives the runtime an opportunity to execute independent branches concurrently.

---

## 7.5 Graph Introspection

A composed Runnable can expose its graph representation.

Conceptually:

```python
graph = chain.get_graph()
```

The graph can then be inspected or rendered.

The important distinction is:

```text
Runnable
   │
   ├── execution interface
   │
   └── structural representation
             │
             ▼
          Graph
```

The graph is therefore a **representation of the Runnable composition**, not a separate orchestration system.

---

## 7.6 Runnable Graph ≠ LangGraph

This distinction is extremely important.

The names are similar, but their purposes differ.

### Runnable Graph

Primarily represents composition:

```text
A → B → C
```

It is fundamentally a description of a Runnable execution pipeline.

### LangGraph

Provides an explicit stateful orchestration runtime:

```text
       ┌───────┐
       │   A   │
       └───┬───┘
           │
           ▼
       ┌───────┐
       │   B   │
       └───┬───┘
           │
      ┌────▼────┐
      │ Decision│
      └─┬─────┬─┘
        │     │
        ▼     ▼
        C     D
        │     │
        └──┬──┘
           │
           ▼
        persist
```

LangGraph explicitly models:

- state
- transitions
- cycles
- persistence
- interrupts
- resumability

A Runnable graph by itself should not be treated as a replacement for LangGraph.

---

## 7.7 Why Graph Introspection Matters

Suppose your production pipeline eventually becomes:

```text
Request
   │
   ▼
Query preprocessing
   │
   ├────────► Retriever A
   │
   ├────────► Retriever B
   │
   └────────► Query classifier
                 │
                 ▼
             Aggregation
                 │
                 ▼
               Model
                 │
                 ▼
              Parser
```

Looking at the source code alone can become difficult.

A graph gives you a structural view:

```text
             ┌──► A ──┐
             │        │
Input ───────┼──► B ──┼──► Aggregate ──► Model ──► Parser
             │        │
             └──► C ──┘
```

This becomes particularly valuable when debugging latency.

You can ask:

```text
Which branch is slow?
Where does parallelism exist?
Where does execution converge?
Which component dominates the critical path?
```

---

## 7.8 Graphs and Observability

There is also a natural relationship between graph structure and tracing.

Conceptually:

```text
Graph structure
      │
      ▼
Execution
      │
      ▼
Run hierarchy
      │
      ▼
Tracing / Observability
```

For example:

```text
Root Run
│
├── Retriever Run
├── Prompt Run
├── Model Run
└── Parser Run
```

This allows observability systems to associate runtime information with individual components.

You can therefore reason about:

- latency per node
- token usage per model call
- errors per component
- retries
- nested execution

This becomes important when we reach **Callbacks and LangSmith**.

---

## 7.9 Graphs and Optimization

The graph also gives us a useful way to reason about performance.

For:

```text
A → B → C
```

the critical path contains all three operations:

```text
T_total ≈ T_A + T_B + T_C
```

For:

```text
      ┌── B ──┐
A ────┤       ├── D
      └── C ──┘
```

the parallel portion behaves approximately as:

```text
T_total ≈ T_A + max(T_B, T_C) + T_D
```

assuming the runtime and downstream systems permit the parallelism.

Thus graph structure isn't merely visual—it helps reason about **latency and concurrency**.

---

## 7.10 Where Runnable Graphs Stop Being Enough

A Runnable graph is excellent for **dataflow composition**.

It becomes less appropriate when your requirements become state-machine-like:

```text
execute
   │
   ▼
inspect result
   │
   ├── success ──► finish
   │
   └── failure
          │
          ▼
       retry
          │
          ▼
       execute
```

or:

```text
execute
   │
   ▼
human approval
   │
   ├── approve ──► continue
   │
   └── reject ───► stop
```

These require explicit state and lifecycle semantics.

That's the architectural boundary where **LangGraph becomes the more appropriate abstraction**.

---

## Quick Recall

| Concept | Meaning |
|---|---|
| Runnable Graph | Structural representation of composed Runnables |
| Node | A Runnable/component in the graph |
| Edge | Data/execution relationship between components |
| Sequence | Linear dataflow |
| Parallel | Independent branches |
| `get_graph()` | Inspect the Runnable's graph structure |
| Runnable Graph | Composition/dataflow representation |
| LangGraph | Stateful orchestration/runtime |

### Key takeaway

> **Runnable composition gives LangChain an executable dataflow graph; graph introspection exposes that structure so we can reason about execution, performance, and observability.**

---
