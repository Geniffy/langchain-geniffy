<p align="center">
  <a href="https://geniffy.com"><img src="https://geniffy.com/brand/geniffy-lockup-ink.png" alt="Geniffy" height="44"></a>
</p>

# Geniffy for LangChain

Give a LangChain agent a memory of each of your users. Add one piece of middleware, and before every call the
model gets the user's briefing: where things stand, what is due, the rules that apply, what happened, and what is
known that bears on their message. When the agent has its answer, the run is saved, tool calls included, into one
memory for the whole conversation. When nothing is known, the model is told so, and says so instead of guessing.

[![CI](https://github.com/Geniffy/langchain-geniffy/actions/workflows/ci.yml/badge.svg)](https://github.com/Geniffy/langchain-geniffy/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/langchain-geniffy)](https://pypi.org/project/langchain-geniffy/)
[![Docs](https://img.shields.io/badge/docs-docs.geniffy.com-1A1814)](https://docs.geniffy.com/integrations/langchain)

```bash
pip install langchain-geniffy
```

Set `GENIFFY_API_KEY` from **API keys** in the [Geniffy app](https://geniffy.com/app). Keep it on your server.

## The middleware

```python
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_geniffy import GeniffyMemory
from langgraph.checkpoint.memory import InMemorySaver


@dataclass
class User:
    id: str


agent = create_agent(
    "anthropic:claude-opus-5-5",
    system_prompt="You are a helpful assistant.",
    middleware=[GeniffyMemory(space=lambda runtime: f"user_{runtime.context.id}")],
    context_schema=User,
    checkpointer=InMemorySaver(),                 # or your database's checkpointer
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Who signs the Lumen renewal?"}]},
    {"configurable": {"thread_id": chat.id}},     # the conversation
    context=User(id=user.id),                     # from your own sign-in
)
```

- **Before each model call**, the user's briefing goes in the system prompt, after your own: where things stand,
  what is due or was promised, the rules that apply, what happened, then what is known that bears on their last
  message, each line dated. In a run with tools, Geniffy is asked once, not once per step.
- **When the agent has its answer**, the run is saved to that user's space: what they said, each tool the agent
  called and what it returned, and the answer. A conversation is one memory however long it gets, and each run
  sends only what is new. Where saving got to is kept in the agent's state, so with a checkpointer nothing is sent
  twice, across restarts and on every server, and a save that fails goes with the next run.
- **If Geniffy can't be reached**, the agent goes on without memory, and `on_error` hears about it.

Options:

- `project`: the project the agent works on, a string or a function of the runtime. The briefing keeps to it, and
  what is saved is labelled with it.
- `session`: the conversation, a string or a function of the runtime, for an app that keeps the history itself
  instead of in a checkpointer. With neither, each run is a memory of its own.
- `briefing=False` reads only what bears on the user's last message; `budget_chars` is the most the briefing adds to
  the prompt (6,000 characters unless you say).
- `remember=False` to only read, and `instructions` to change what the model is told about the memory.
- `client` / `async_client` to bring your own `geniffy.Geniffy` / `geniffy.AsyncGeniffy`. Async agents (`ainvoke`,
  `astream`) use the async client.

An account without the briefing switched on is given what bears on the user's last message instead, and the
briefing is tried again ten minutes later.

## Tools

To let the agent decide when to look something up or save it:

```python
from langchain_geniffy import geniffy_tools

agent = create_agent(
    "anthropic:claude-opus-5-5",
    tools=geniffy_tools(space=lambda runtime: f"user_{runtime.context.id}"),   # recall and remember
    context_schema=User,
)
```

The tools read the user from the runtime, so the model never sees or chooses whose memory it reads.

## Retriever

```python
from langchain_geniffy import GeniffyRetriever

docs = GeniffyRetriever(space="user_1042", k=5).invoke("the Lumen renewal")
docs[0].metadata    # {"id": ..., "kind": "people", "about": "Priya Nair", "said_at": "...", "source": "Call with Priya", ...}
```

## One space per user

`space` is required: the user this is for, as a string or a function of the agent's runtime. Each space is a
memory of its own, and nothing else can read it. `space=None` is your own memory, never your users' data. A
blank space, or None from your function, is refused, so a user with no id never lands in your own memory. When
a user deletes their account, forget them with `Geniffy().forget_space(...)` from the
[geniffy](https://pypi.org/project/geniffy/) SDK.

## Develop

```bash
pip install -e ".[test]" && pytest     # through LangChain's own create_agent, with a scripted model and the
                                       # real geniffy client against a stand-in for the API
```

## Security

Report a vulnerability to ops@geniffy.com, not in a public issue. See the
[security policy](https://github.com/Geniffy/.github/blob/main/SECURITY.md).
