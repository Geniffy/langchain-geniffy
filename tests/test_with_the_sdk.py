"""GeniffyMemory with the real geniffy client, its calls answered by a stand-in for the API: what goes over the
wire for the briefing, for a conversation saved as it goes, and for an account without the briefing switched on."""
from __future__ import annotations

import asyncio
import json
from typing import Any, List

import httpx
from geniffy import AsyncGeniffy, Geniffy
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from langchain_geniffy import GeniffyMemory, __version__
from test_langchain_geniffy import BRIEFING, CONTEXT, Scripted, User, per_user

SOURCE = {"id": "a" * 32, "kind": "note", "title": "Who signs the Lumen renewal?", "status": "reading"}


class Api:
    """The API, as far as the middleware reaches it: the briefing (or not switched on), context and adding."""

    def __init__(self, briefing_on: bool = True):
        self.briefing_on = briefing_on
        self.calls: List[Any] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        self.calls.append((request.url.path, request.headers.get("x-geniffy-space"), body))
        assert f"langchain-geniffy/{__version__}" in request.headers["user-agent"]
        if request.url.path == "/v1/briefing":
            if not self.briefing_on:
                return httpx.Response(404, json={"error": {"code": "not_switched_on", "message": (
                    "The briefing, where things stand, episodes, lessons and intentions aren't switched on for "
                    "this memory yet.")}})
            return httpx.Response(200, json={"project": body.get("project"), "briefing": BRIEFING, "now": [],
                                             "due": [], "lessons": [], "episodes": [], "memories": []})
        if request.url.path == "/v1/context":
            return httpx.Response(200, json={"context": CONTEXT, "memories": [], "found": True})
        if request.url.path == "/v1/memories":
            return httpx.Response(201, json={"source": SOURCE})
        return httpx.Response(404, json={"error": {"code": "not_found", "message": "No such door."}})


def made(api: Api) -> GeniffyMemory:
    client = Geniffy("gnf_test", base_url="https://api.test", integration=f"langchain-geniffy/{__version__}",
                     http_client=httpx.Client(transport=httpx.MockTransport(api)), max_retries=0)
    return GeniffyMemory(space=per_user, project="lumen", client=client)


def test_the_briefing_and_the_session_as_they_go_over_the_wire():
    api, saver = Api(), InMemorySaver()
    memory = made(api)
    thread = {"configurable": {"thread_id": "chat-7"}}
    for question, answer in (("Who signs the Lumen renewal?", "Priya Nair signs it."), ("And when?", "In March.")):
        agent = create_agent(Scripted(replies=[answer], seen=[]), context_schema=User, checkpointer=saver,
                             middleware=[memory])
        agent.invoke({"messages": [{"role": "user", "content": question}]}, thread, context=User(id="1042"))
    assert api.calls == [
        ("/v1/briefing", "user_1042",
         {"project": "lumen", "cue": "Who signs the Lumen renewal?", "budget_chars": 6000}),
        ("/v1/memories", "user_1042", {"messages": [{"role": "user", "content": "Who signs the Lumen renewal?"},
                                                    {"role": "assistant", "content": "Priya Nair signs it."}],
                                       "session": "chat-7", "labels": {"project": "lumen"},
                                       "title": "Who signs the Lumen renewal?"}),
        ("/v1/briefing", "user_1042", {"project": "lumen", "cue": "And when?", "budget_chars": 6000}),
        ("/v1/memories", "user_1042", {"messages": [{"role": "user", "content": "And when?"},
                                                    {"role": "assistant", "content": "In March."}],
                                       "session": "chat-7", "labels": {"project": "lumen"}, "title": "And when?"}),
    ]


def test_an_account_without_the_briefing_reads_context_over_the_wire():
    api = Api(briefing_on=False)
    model = Scripted(replies=["Priya Nair signs it."], seen=[])
    agent = create_agent(model, context_schema=User, middleware=[made(api)])
    agent.invoke({"messages": [{"role": "user", "content": "Who signs the Lumen renewal?"}]}, context=User(id="1042"))
    assert [path for path, _, _ in api.calls] == ["/v1/briefing", "/v1/context", "/v1/memories"]
    assert f"<memory>\n{CONTEXT}\n</memory>" in model.seen[0][0].text


def test_the_async_client_over_the_wire():
    api = Api()
    client = AsyncGeniffy("gnf_test", base_url="https://api.test", integration=f"langchain-geniffy/{__version__}",
                          http_client=httpx.AsyncClient(transport=httpx.MockTransport(api)), max_retries=0)
    agent = create_agent(Scripted(replies=["Priya Nair signs it."], seen=[]), context_schema=User,
                         checkpointer=InMemorySaver(),
                         middleware=[GeniffyMemory(space=per_user, async_client=client)])
    asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "Who signs the Lumen renewal?"}]},
                              {"configurable": {"thread_id": "chat-9"}}, context=User(id="1042")))
    assert [(path, body.get("session")) for path, _, body in api.calls] == [("/v1/briefing", None),
                                                                            ("/v1/memories", "chat-9")]
