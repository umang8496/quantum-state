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


# 3. The `Runnable` Protocol

`Runnable` is the **core execution abstraction in modern LangChain**.

If you understand `Runnable` well, most of LCEL becomes straightforward.

## 3.1 What Problem Does `Runnable` Solve?

Different LangChain components naturally perform different operations:

```text
Prompt       : input → messages
Chat Model   : messages → AIMessage
Parser       : AIMessage → Python object
Retriever    : query → Documents
Function     : input → output
```

Without a common interface, composing these components would require custom glue code for every combination.

`Runnable` provides that common contract:

```text
Runnable[Input, Output]

Input ──invoke()──► Output
```

---

## 3.2 The Core Execution Methods

A `Runnable` provides four fundamental execution styles:

| Method | Purpose |
|---|---|
| `invoke()` | Execute one input synchronously |
| `ainvoke()` | Execute one input asynchronously |
| `batch()` | Execute multiple inputs synchronously |
| `abatch()` | Execute multiple inputs asynchronously |

There are corresponding streaming operations:

```text
stream()
astream()
```

So conceptually:

```text
                  Runnable
                     │
       ┌─────────────┼─────────────┐
       │             │             │
    invoke         batch         stream
       │             │             │
    ainvoke        abatch        astream
```

---

## 3.3 Why This Is Powerful

Because different components implement the same contract, they can be composed uniformly.

For example:

```text
Prompt
  │
  ▼
ChatModel
  │
  ▼
Parser
```

Each component is independently a `Runnable`.

Therefore the complete pipeline is itself a `Runnable`:

```text
Runnable
   │
   ├── Prompt
   ├── ChatModel
   └── Parser
```

This is an important property:

> **Composition produces another Runnable.**

That means a small pipeline can become a component of a larger pipeline.

---

## 3.4 LCEL's `|` Operator

This is what enables:

```python
chain = prompt | model | parser
```

Conceptually:

```text
prompt ──► model ──► parser
```

The `|` operator creates a composed Runnable.

Therefore:

```python
chain.invoke(input)
```

executes the complete pipeline.

You don't need to manually write:

```python
prompt_result = prompt.invoke(input)
model_result = model.invoke(prompt_result)
parser_result = parser.invoke(model_result)
```

The composition abstraction handles that execution.

---

## 3.5 Native Python Equivalent

Without LangChain, you might explicitly write:

```python
def process(user_input: str) -> Output:
    prompt = build_prompt(user_input)
    response = call_model(prompt)
    return parse_response(response)
```

With Runnables:

```text
build_prompt
     │
     ▼
call_model
     │
     ▼
parse_response
```

becomes a composable execution graph.

The important benefit isn't saving a few lines of code.

The benefit is that the **same execution semantics** can then be applied to:

- synchronous execution
- asynchronous execution
- batching
- streaming
- callbacks
- tracing
- configuration
- retries
- fallbacks
- concurrency control

---

## 3.6 A Runnable Is More Than a Function

It is tempting to think:

```python
Runnable ≈ Callable
```

That's incomplete.

A Python callable primarily gives you:

```python
output = function(input)
```

A Runnable provides an **execution protocol** around the operation.

Conceptually:

```text
                Runnable
                   │
       ┌───────────┴───────────┐
       │                       │
   Computation             Execution
       │                       │
       │              ┌────────┼────────┐
       │              │        │        │
       │            async    batch   stream
       │
       └───────────────┐
                       ▼
                 input → output
```

This execution protocol is what makes LangChain components interoperable.

---

## 3.7 The Most Important Mental Model

Think of `Runnable` as the **"function + execution runtime contract"**.

```text
Python function

input → output


Runnable

input
  │
  ├── invoke
  ├── async
  ├── batch
  ├── streaming
  ├── configuration
  ├── callbacks
  └── tracing
       │
       ▼
     output
```

That is the foundation upon which **LCEL, model pipelines, retrieval pipelines, and many LangChain components** are built.

### Key takeaway

> **`Runnable` is the common execution protocol that turns individual LangChain components into composable, observable, synchronous/asynchronous, batchable, and streamable building blocks.**
