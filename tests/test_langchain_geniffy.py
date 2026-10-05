"""langchain-geniffy through LangChain's own create_agent, with a chat model that records what it was given
and stand-ins for the Geniffy clients: what the model sees, when an exchange is saved, tool loops, the
tools, the retriever, and a memory that is away."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, List

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_geniffy import GeniffyMemory, GeniffyRetriever, geniffy_tools

CONTEXT = "- Priya Nair signs the Lumen renewal.  [Call with Priya, 1 Oct 2026]"


@dataclass
class User:
    id: str


class Scripted(BaseChatModel):
    """Answers from a script, one reply per call, and keeps every prompt it was given."""

    replies: List[Any]
    seen: List[Any] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "Scripted":
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.seen.append(list(messages))
        reply = self.replies.pop(0)
        return ChatResult(generations=[ChatGeneration(message=reply if isinstance(reply, AIMessage) else AIMessage(reply))])


class Mem:
    def __init__(self, log: dict, space: Any, fail: bool = False):
        self.log, self.space, self.fail = log, space, fail
        self.memories = self

    def context(self, question: str) -> str:
        if self.fail:
            raise RuntimeError("Geniffy is away")
        self.log["context"].append((self.space, question))
        return CONTEXT

    def add(self, text: Any = None, *, messages: Any = None) -> dict:
        if self.fail:
            raise RuntimeError("Geniffy is away")
        self.log["add"].append((self.space, text if messages is None else messages))
        return {"id": "src_1"}

    def search(self, q: str, *, limit: int = 10) -> list:
        from geniffy import Memory
        self.log["search"].append((self.space, q, limit))
        return [Memory.from_json({"id": 7, "text": "Priya Nair signs the Lumen renewal.", "kind": "people",
                                  "about": "Priya Nair", "said_at": "2026-10-01T09:00:00+00:00",
                                  "source": {"id": "a" * 32, "kind": "note", "title": "Call with Priya"}})]


class AsyncMem(Mem):
    async def context(self, question: str) -> str:  # type: ignore[override]
        return Mem.context(self, question)

    async def add(self, text: Any = None, *, messages: Any = None) -> dict:  # type: ignore[override]
        return Mem.add(self, text, messages=messages)


class Client:
    def __init__(self, mem_class: type = Mem, fail: bool = False):
        self.log = {"context": [], "add": [], "search": []}
        self.mem_class, self.fail = mem_class, fail

    def space(self, name: str) -> Mem:
        return self.mem_class(self.log, name, self.fail)


def per_user(runtime: Any) -> str:
    return f"user_{runtime.context.id}"


def ask(agent: Any, question: str = "Who signs the Lumen renewal?") -> Any:
    return agent.invoke({"messages": [{"role": "user", "content": question}]}, context=User(id="1042"))


def test_a_space_is_required():
    with pytest.raises(TypeError, match="needs a space"):
        GeniffyMemory()
    with pytest.raises(TypeError, match="needs a space"):
        geniffy_tools()


def test_a_blank_space_or_a_missing_user_is_refused_never_your_own_memory():
    """A blank space, or None from a function of the runtime, is what a missing user id looks like: read as
    no space, that user's words would land in your own memory. Each is refused, and nothing is asked."""
    for blank in ("", "   "):
        with pytest.raises(ValueError, match="blank"):
            GeniffyMemory(space=blank)
        with pytest.raises(ValueError, match="blank"):
            geniffy_tools(space=blank)
        with pytest.raises(ValueError, match="blank"):
            GeniffyRetriever(space=blank, client=Client())
    with pytest.raises(TypeError, match="string"):
        GeniffyMemory(space=1042)

    for missing, refusal in ((lambda runtime: None, "returned None"), (lambda runtime: " ", "blank"),
                             (lambda runtime: 1042, "must return a string")):
        client = Client()
        model = Scripted(replies=["Hello."], seen=[])
        agent = create_agent(model, context_schema=User, middleware=[GeniffyMemory(space=missing, client=client)])
        with pytest.raises((ValueError, TypeError), match=refusal):
            ask(agent)
        assert client.log == {"context": [], "add": [], "search": []}

    class Own(Mem):
        def space(self, name: str) -> Mem:
            raise AssertionError("space=None must not pick a space")

    own = Own({"context": [], "add": [], "search": []}, None)
    model = Scripted(replies=["Hello."], seen=[])
    agent = create_agent(model, context_schema=User, middleware=[GeniffyMemory(space=None, client=own)])
    ask(agent)
    assert own.log["context"] == [(None, "Who signs the Lumen renewal?")], "space=None is your own memory"


def test_the_model_is_given_what_is_known_after_the_apps_own_prompt_and_the_exchange_is_saved():
    client = Client()
    model = Scripted(replies=["Priya Nair signs it."], seen=[])
    agent = create_agent(model, system_prompt="You are a helpful assistant.", context_schema=User,
                         middleware=[GeniffyMemory(space=per_user, client=client)])
    result = ask(agent)
    assert result["messages"][-1].text == "Priya Nair signs it."
    system = model.seen[0][0]
    assert system.type == "system" and system.text.startswith("You are a helpful assistant.\n\n")
    assert f"<memory>\n{CONTEXT}\n</memory>" in system.text
    assert client.log["context"] == [("user_1042", "Who signs the Lumen renewal?")]
    assert client.log["add"] == [("user_1042", [{"role": "user", "content": "Who signs the Lumen renewal?"},
                                               {"role": "assistant", "content": "Priya Nair signs it."}])]


def test_async_agents_use_the_async_client():
    client = Client(AsyncMem)
    model = Scripted(replies=["Priya Nair signs it."], seen=[])
    agent = create_agent(model, context_schema=User,
                         middleware=[GeniffyMemory(space=per_user, async_client=client)])
    result = asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "Who signs the Lumen renewal?"}]},
                                       context=User(id="1042")))
    assert result["messages"][-1].text == "Priya Nair signs it."
    assert client.log["context"] == [("user_1042", "Who signs the Lumen renewal?")]
    assert len(client.log["add"]) == 1


def test_a_tool_loop_asks_geniffy_once_and_the_tools_read_the_user_from_the_runtime():
    client = Client()
    model = Scripted(seen=[], replies=[
        AIMessage("", tool_calls=[{"name": "recall", "args": {"query": "Lumen renewal"}, "id": "c1"}]),
        AIMessage("", tool_calls=[{"name": "remember", "args": {"text": "I prefer mornings."}, "id": "c2"}]),
        "Priya Nair signs it, and I'll remember you prefer mornings.",
    ])
    agent = create_agent(model, tools=geniffy_tools(space=per_user, client=client), context_schema=User,
                         middleware=[GeniffyMemory(space=per_user, client=client)])
    result = ask(agent)
    assert result["messages"][-1].text.startswith("Priya Nair signs it")
    tool_outputs = [m.text for m in result["messages"] if m.type == "tool"]
    assert tool_outputs == [CONTEXT, "Saved."]
    assert client.log["context"] == [("user_1042", "Who signs the Lumen renewal?"), ("user_1042", "Lumen renewal")], \
        "the middleware asked once for three model calls; the recall tool asked once"
    saved = [a for a in client.log["add"] if a[1] == "I prefer mornings."]
    assert saved == [("user_1042", "I prefer mornings.")]
    exchanges = [a for a in client.log["add"] if isinstance(a[1], list)]
    assert len(exchanges) == 1 and exchanges[0][1][1]["content"].startswith("Priya Nair signs it")


def test_a_memory_that_is_away_never_breaks_the_answer():
    errors = []
    model = Scripted(replies=["Hello!"], seen=[])
    agent = create_agent(model, context_schema=User,
                         middleware=[GeniffyMemory(space=per_user, client=Client(fail=True), on_error=errors.append)])
    assert ask(agent, "Hello there, how are you?")["messages"][-1].text == "Hello!"
    assert [m.type for m in model.seen[0]] == ["human"], "nothing added when the memory is away"
    assert [str(e) for e in errors] == ["Geniffy is away", "Geniffy is away"]


def test_remember_false_only_reads():
    client = Client()
    agent = create_agent(Scripted(replies=["Hi."], seen=[]), context_schema=User,
                         middleware=[GeniffyMemory(space=per_user, client=client, remember=False)])
    ask(agent, "Hello there, how are you?")
    assert client.log["add"] == [] and len(client.log["context"]) == 1


def test_the_retriever_returns_dated_documents_with_their_source():
    client = Client()
    docs = GeniffyRetriever(space="user_1042", k=5, client=client).invoke("the Lumen renewal")
    assert docs[0].page_content == "Priya Nair signs the Lumen renewal."
    assert docs[0].metadata == {"id": 7, "kind": "people", "about": "Priya Nair", "said_at": "2026-10-01T09:00:00+00:00",
                                "learned_at": None, "source": "Call with Priya"}
    assert client.log["search"] == [("user_1042", "the Lumen renewal", 5)]
