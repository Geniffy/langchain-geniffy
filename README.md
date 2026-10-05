<p align="center">
  <a href="https://geniffy.com"><img src="https://geniffy.com/brand/geniffy-lockup-ink.png" alt="Geniffy" height="44"></a>
</p>

# Geniffy for LangChain

Give a LangChain agent a memory of each of your users. Add one piece of middleware, and before every call the
model is told what is known about the user, each line with where it came from; when the agent has its answer,
the exchange is saved. When nothing is known, the model is told so, and says so instead of guessing.

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


@dataclass
class User:
    id: str


agent = create_agent(
    "anthropic:claude-opus-5-5",
    system_prompt="You are a helpful assistant.",
    middleware=[GeniffyMemory(space=lambda runtime: f"user_{runtime.context.id}")],
    context_schema=User,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Who signs the Lumen renewal?"}]},
    context=User(id=user.id),                     # from your own sign-in
)
```

- **Before each model call**, what is known that bears on the user's last message goes in the system prompt,
  after your own. In a run with tools, Geniffy is asked once, not once per step.
- **When the agent has its answer**, the exchange is saved to that user's space.
- **If Geniffy can't be reached**, the agent goes on without memory, and `on_error` hears about it.

Options: `remember=False` to only read, `instructions` to change what the model is told about the memory, and
`client` / `async_client` to bring your own `geniffy.Geniffy` / `geniffy.AsyncGeniffy`. Async agents
(`ainvoke`, `astream`) use the async client.

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
pip install -e ".[test]" && pytest     # through LangChain's own create_agent, with a scripted model
```

## Security

Report a vulnerability to ops@geniffy.com, not in a public issue. See the
[security policy](https://github.com/Geniffy/.github/blob/main/SECURITY.md).
