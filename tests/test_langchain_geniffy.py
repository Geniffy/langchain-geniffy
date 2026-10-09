"""langchain-geniffy through LangChain's own create_agent, with a chat model that records what it was given
and stand-ins for the Geniffy clients: what the model sees, how a conversation is saved as it goes (tool calls
included, never twice, across restarts), tool loops, the tools, the retriever, an account without the briefing,
and a memory that is away."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, List

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from langchain_geniffy import GeniffyMemory, GeniffyRetriever, geniffy_tools
from langchain_geniffy._core import as_sent

CONTEXT = "- Priya Nair signs the Lumen renewal.  [Call with Priya, 1 Oct 2026]"
BRIEFING = ("Where things stand (lumen), as of 2 Oct:\n- Next: send Priya the renewal\n"
            "What is known (facts):\n- [2026-10-01] Priya Nair signs the Lumen renewal.")


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


class Status(Exception):
    """An error from the API, with its status, as the geniffy SDK raises them."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class Mem:
    """One space of the stand-in client. What it was asked is kept in the client's log."""

    def __init__(self, client: "Client", space: Any):
        self.client, self.log, self.space = client, client.log, space
        self.memories = self

    def briefing(self, *, project: Any = None, cue: str = "", budget_chars: int = 6000) -> str:
        if self.client.fail:
            raise RuntimeError("Geniffy is away")
        self.log["briefing"].append((self.space, project, cue))
        if isinstance(self.client.briefing, Exception):
            raise self.client.briefing
        return self.client.briefing

    def context(self, question: str) -> str:
        if self.client.fail:
            raise RuntimeError("Geniffy is away")
        self.log["context"].append((self.space, question))
        return CONTEXT

    def add(self, text: Any = None, *, messages: Any = None, session: Any = None, title: Any = None,
            labels: Any = None) -> dict:
        if self.client.fail:
            raise RuntimeError("Geniffy is away")
        if messages is None:
            self.log["add"].append((self.space, text))
            return {"id": "src_1"}
        if self.client.saves_fail:
            self.client.saves_fail -= 1
            raise Status(502, "Your memory didn't answer. Try again in a minute.")
        self.log["session"].append({"space": self.space, "session": session, "title": title, "labels": labels,
                                    "messages": messages})
        return {"id": "src_1"}

    def search(self, q: str, *, limit: int = 10) -> list:
        from geniffy import Memory
        self.log["search"].append((self.space, q, limit))
        return [Memory.from_json({"id": 7, "text": "Priya Nair signs the Lumen renewal.", "kind": "people",
                                  "about": "Priya Nair", "said_at": "2026-10-01T09:00:00+00:00",
                                  "source": {"id": "a" * 32, "kind": "note", "title": "Call with Priya"}})]


class AsyncMem(Mem):
    async def briefing(self, **kwargs: Any) -> str:  # type: ignore[override]
        return Mem.briefing(self, **kwargs)

    async def context(self, question: str) -> str:  # type: ignore[override]
        return Mem.context(self, question)

    async def add(self, text: Any = None, **kwargs: Any) -> dict:  # type: ignore[override]
        return Mem.add(self, text, **kwargs)


class Client:
    """A stand-in for geniffy.Geniffy. briefing is what the briefing says, or the error it raises; saves_fail is
    how many of the next saves fail."""

    def __init__(self, mem_class: type = Mem, fail: bool = False, briefing: Any = BRIEFING, saves_fail: int = 0):
        self.log: dict = {"briefing": [], "context": [], "add": [], "session": [], "search": []}
        self.mem_class, self.fail, self.briefing, self.saves_fail = mem_class, fail, briefing, saves_fail

    def space(self, name: str) -> Mem:
        return self.mem_class(self, name)


NOTHING_ASKED = {"briefing": [], "context": [], "add": [], "session": [], "search": []}


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
        assert client.log == NOTHING_ASKED

    class Own(Mem):
        def space(self, name: str) -> Mem:
            raise AssertionError("space=None must not pick a space")

    own = Own(Client(), None)
    model = Scripted(replies=["Hello."], seen=[])
    agent = create_agent(model, context_schema=User, middleware=[GeniffyMemory(space=None, client=own)])
    ask(agent)
    assert own.log["briefing"] == [(None, None, "Who signs the Lumen renewal?")], "space=None is your own memory"
    assert [s["space"] for s in own.log["session"]] == [None]


def test_the_model_is_given_the_briefing_after_the_apps_own_prompt_and_the_run_is_saved():
    client = Client()
    model = Scripted(replies=["Priya Nair signs it."], seen=[])
    agent = create_agent(model, system_prompt="You are a helpful assistant.", context_schema=User,
                         middleware=[GeniffyMemory(space=per_user, client=client)])
    result = ask(agent)
    assert result["messages"][-1].text == "Priya Nair signs it."
    system = model.seen[0][0]
    assert system.type == "system" and system.text.startswith("You are a helpful assistant.\n\n")
    assert f"<memory>\n{BRIEFING}\n</memory>" in system.text
    assert client.log["briefing"] == [("user_1042", None, "Who signs the Lumen renewal?")]
    assert client.log["context"] == [], "the briefing holds what is known that bears on the question"
    [saved] = client.log["session"]
    assert saved["space"] == "user_1042" and saved["session"].startswith("run-"), \
        "with no checkpointer and no session, a run is a memory of its own"
    assert saved["title"] == "Who signs the Lumen renewal?" and saved["labels"] is None
    assert saved["messages"] == [{"role": "user", "content": "Who signs the Lumen renewal?"},
                                 {"role": "assistant", "content": "Priya Nair signs it."}]
    assert "geniffy_saved" not in result, "where saving got to is the middleware's own, not in the output"


def test_async_agents_use_the_async_client():
    client = Client(AsyncMem)
    model = Scripted(replies=["Priya Nair signs it."], seen=[])
    agent = create_agent(model, context_schema=User,
                         middleware=[GeniffyMemory(space=per_user, async_client=client)])
    result = asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "Who signs the Lumen renewal?"}]},
                                       context=User(id="1042")))
    assert result["messages"][-1].text == "Priya Nair signs it."
    assert client.log["briefing"] == [("user_1042", None, "Who signs the Lumen renewal?")]
    assert len(client.log["session"]) == 1


def test_a_tool_loop_reads_the_briefing_once_and_saves_the_whole_run_with_its_tool_calls():
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
    assert client.log["briefing"] == [("user_1042", None, "Who signs the Lumen renewal?")], \
        "the middleware asked once for three model calls"
    assert client.log["context"] == [("user_1042", "Lumen renewal")], "the recall tool asked once"
    assert client.log["add"] == [("user_1042", "I prefer mornings.")]
    [saved] = client.log["session"]
    assert saved["messages"] == [
        {"role": "user", "content": "Who signs the Lumen renewal?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"name": "recall", "args": {"query": "Lumen renewal"}, "id": "c1"}]},
        {"role": "tool", "content": CONTEXT, "name": "recall", "tool_call_id": "c1"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"name": "remember", "args": {"text": "I prefer mornings."}, "id": "c2"}]},
        {"role": "tool", "content": "Saved.", "name": "remember", "tool_call_id": "c2"},
        {"role": "assistant", "content": "Priya Nair signs it, and I'll remember you prefer mornings."},
    ], "what the agent did is how Geniffy learns what happened"


def test_a_conversation_with_a_checkpointer_is_one_memory_and_no_turn_is_sent_twice_even_after_a_restart():
    """Where saving got to is kept in the agent's state, so the checkpointer keeps it with the thread: a second
    process (a restart, or another server behind the same database) sends only what is new."""
    client, saver = Client(), InMemorySaver()
    thread = {"configurable": {"thread_id": "chat-7"}}

    def turn(question: str, answer: str) -> None:
        agent = create_agent(Scripted(replies=[answer], seen=[]), context_schema=User, checkpointer=saver,
                             middleware=[GeniffyMemory(space=per_user, client=client)])   # a fresh process each turn
        agent.invoke({"messages": [{"role": "user", "content": question}]}, thread, context=User(id="1042"))

    turn("Who signs the Lumen renewal?", "Priya Nair signs it.")
    turn("And when is it due?", "In March.")
    turn("Thanks.", "Any time.")
    assert [s["session"] for s in client.log["session"]] == ["chat-7", "chat-7", "chat-7"], "one memory for the thread"
    assert [[m["content"] for m in s["messages"]] for s in client.log["session"]] == [
        ["Who signs the Lumen renewal?", "Priya Nair signs it."],
        ["And when is it due?", "In March."],
        ["Thanks.", "Any time."],
    ]


def test_a_save_that_fails_goes_with_the_next_run():
    errors: list = []
    client, saver = Client(saves_fail=1), InMemorySaver()
    thread = {"configurable": {"thread_id": "chat-8"}}
    agent = create_agent(Scripted(replies=["Priya Nair signs it.", "In March."], seen=[]), context_schema=User,
                         checkpointer=saver,
                         middleware=[GeniffyMemory(space=per_user, client=client, on_error=errors.append)])
    for question in ("Who signs the Lumen renewal?", "And when is it due?"):
        agent.invoke({"messages": [{"role": "user", "content": question}]}, thread, context=User(id="1042"))
    assert [e.status for e in errors] == [502]
    [saved] = client.log["session"]
    assert [m["content"] for m in saved["messages"]] == [
        "Who signs the Lumen renewal?", "Priya Nair signs it.", "And when is it due?", "In March."]


def test_a_project_and_a_session_from_the_runtime_and_a_history_your_app_keeps():
    """Without a checkpointer your app passes the conversation each time: only this turn is sent, under your id."""
    client = Client()
    memory = GeniffyMemory(space=per_user, project="lumen", session=lambda runtime: f"chat-{runtime.context.id}",
                           client=client)
    agent = create_agent(Scripted(replies=["Priya Nair signs it.", "In March."], seen=[]), context_schema=User,
                         middleware=[memory])
    first = agent.invoke({"messages": [{"role": "user", "content": "Who signs the Lumen renewal?"}]},
                         context=User(id="1042"))
    agent.invoke({"messages": first["messages"] + [{"role": "user", "content": "And when is it due?"}]},
                 context=User(id="1042"))
    assert [b[1] for b in client.log["briefing"]] == ["lumen", "lumen"]
    assert [(s["session"], s["labels"]) for s in client.log["session"]] == [
        ("chat-1042", {"project": "lumen"}), ("chat-1042", {"project": "lumen"})]
    assert [[m["content"] for m in s["messages"]] for s in client.log["session"]] == [
        ["Who signs the Lumen renewal?", "Priya Nair signs it."], ["And when is it due?", "In March."]]


def test_without_the_briefing_switched_on_it_reads_what_is_known_and_tries_again_later():
    client = Client(briefing=Status(404, "The briefing isn't switched on for this memory yet."))
    memory = GeniffyMemory(space=per_user, client=client, on_error=lambda e: pytest.fail(f"not an error: {e}"))
    model = Scripted(replies=["Priya Nair signs it.", "Hello."], seen=[])
    agent = create_agent(model, context_schema=User, middleware=[memory])
    ask(agent)
    ask(agent, "Hello there, how are you?")
    assert f"<memory>\n{CONTEXT}\n</memory>" in model.seen[0][0].text
    assert len(client.log["briefing"]) == 1, "asked once, then not for ten minutes"
    assert [q for _, q in client.log["context"]] == ["Who signs the Lumen renewal?", "Hello there, how are you?"]

    client.briefing, memory._off_until = BRIEFING, 0.0       # ten minutes on, and it is switched on
    agent = create_agent(Scripted(replies=["Priya Nair signs it."], seen=[]), context_schema=User, middleware=[memory])
    ask(agent, "And when is it due?")
    assert client.log["briefing"][-1] == ("user_1042", None, "And when is it due?")


def test_a_briefing_that_fails_another_way_is_an_error_not_a_reason_to_read_something_else():
    errors: list = []
    client = Client(briefing=Status(502, "Your memory didn't answer. Try again in a minute."))
    model = Scripted(replies=["Hello."], seen=[])
    agent = create_agent(model, context_schema=User,
                         middleware=[GeniffyMemory(space=per_user, client=client, on_error=errors.append)])
    ask(agent)
    assert [e.status for e in errors] == [502] and client.log["context"] == []
    assert [m.type for m in model.seen[0]] == ["human"]


def test_an_empty_briefing_still_says_nothing_is_remembered_yet():
    model = Scripted(replies=["I don't know yet."], seen=[])
    agent = create_agent(model, context_schema=User,
                         middleware=[GeniffyMemory(space=per_user, client=Client(briefing=""))])
    ask(agent, "What is my name?")
    assert "<memory>\nNothing is remembered about this user yet.\n</memory>" in model.seen[0][0].text


def test_briefing_false_reads_only_what_bears_on_the_question():
    client = Client()
    model = Scripted(replies=["Priya Nair signs it."], seen=[])
    agent = create_agent(model, context_schema=User,
                         middleware=[GeniffyMemory(space=per_user, client=client, briefing=False)])
    ask(agent)
    assert client.log["briefing"] == [] and client.log["context"] == [("user_1042", "Who signs the Lumen renewal?")]
    assert f"<memory>\n{CONTEXT}\n</memory>" in model.seen[0][0].text


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
    assert client.log["session"] == [] and len(client.log["briefing"]) == 1


def test_messages_are_sent_as_geniffy_reads_them_and_no_larger_than_it_keeps():
    blocks = AIMessage(content=[{"type": "text", "text": "Looking that up."},
                                {"type": "tool_use", "id": "c9", "name": "fetch", "input": {"url": "https://x.test"}}],
                       tool_calls=[{"name": "fetch", "args": {"url": "https://x.test"}, "id": "c9"}])
    assert as_sent(blocks) == {"role": "assistant", "content": "Looking that up.",
                               "tool_calls": [{"name": "fetch", "args": {"url": "https://x.test"}, "id": "c9"}]}, \
        "a call is sent once, not once as a block and again as a tool call"
    page = as_sent(ToolMessage(content="x" * 10_000, tool_call_id="c9", name="fetch"))
    assert len(page["content"]) == 4_000
    big = as_sent(AIMessage("", tool_calls=[{"name": "write", "args": {"text": "y" * 9_000}, "id": "c10"}]))
    assert isinstance(big["tool_calls"][0]["args"], str) and len(big["tool_calls"][0]["args"]) == 2_000
    assert as_sent(SystemMessage("You are a helpful assistant.")) is None
    assert as_sent(HumanMessage("Hi", name="Ana")) == {"role": "user", "content": "Hi", "name": "Ana"}


def test_a_run_of_more_than_500_messages_is_sent_500_at_a_time_and_a_failure_keeps_what_was_sent():
    class Runtime:
        context = User(id="1042")

    history = [HumanMessage("Go through the inbox.", id="m0")]
    for i in range(1, 1001):
        history.append(AIMessage(f"Item {i} done.", id=f"m{i}"))
    client = Client()
    memory = GeniffyMemory(space=per_user, session="inbox", client=client)
    assert memory.after_agent({"messages": history}, Runtime()) == {"geniffy_saved": "m1000"}
    assert [len(s["messages"]) for s in client.log["session"]] == [500, 500, 1]

    client = Client(saves_fail=0)
    memory = GeniffyMemory(space=per_user, session="inbox", client=client, on_error=lambda e: None)
    calls = {"n": 0}
    real_add = Mem.add

    def second_fails(self: Mem, text: Any = None, **kwargs: Any) -> dict:
        calls["n"] += 1
        if calls["n"] == 2:
            raise Status(502, "Your memory didn't answer.")
        return real_add(self, text, **kwargs)

    Mem.add = second_fails  # type: ignore[method-assign]
    try:
        assert memory.after_agent({"messages": history}, Runtime()) == {"geniffy_saved": "m499"}, \
            "the next run starts after what was sent"
    finally:
        Mem.add = real_add  # type: ignore[method-assign]


def test_the_retriever_returns_dated_documents_with_their_source():
    client = Client()
    docs = GeniffyRetriever(space="user_1042", k=5, client=client).invoke("the Lumen renewal")
    assert docs[0].page_content == "Priya Nair signs the Lumen renewal."
    assert docs[0].metadata == {"id": 7, "kind": "people", "about": "Priya Nair", "said_at": "2026-10-01T09:00:00+00:00",
                                "learned_at": None, "source": "Call with Priya"}
    assert client.log["search"] == [("user_1042", "the Lumen renewal", 5)]
