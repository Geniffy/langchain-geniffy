"""recall and remember, for an agent that decides for itself when to look something up or save it."""
from __future__ import annotations

from typing import Any, List

from langchain.tools import ToolRuntime, tool

from ._core import Space, bound, require_space, space_for


def geniffy_tools(*, space: Space = ..., client: Any = None, async_client: Any = None) -> List[Any]:
    """[recall, remember], reading the user from the agent's runtime, so the model never sees or chooses
    whose memory it reads.

        agent = create_agent(model, tools=geniffy_tools(space=lambda rt: f"user_{rt.context.id}"),
                             context_schema=User)
    """
    require_space(space)
    clients: dict = {"sync": client, "async": async_client}

    def sync_mem(runtime: Any) -> Any:
        if clients["sync"] is None:
            from geniffy import Geniffy
            clients["sync"] = Geniffy()
        return bound(clients["sync"], space_for(space, runtime))

    def async_mem(runtime: Any) -> Any:
        if clients["async"] is None:
            from geniffy import AsyncGeniffy
            clients["async"] = AsyncGeniffy()
        return bound(clients["async"], space_for(space, runtime))

    async def arecall(query: str, runtime: ToolRuntime[Any]) -> str:
        return await async_mem(runtime).context(query)

    async def aremember(text: str, runtime: ToolRuntime[Any]) -> str:
        await async_mem(runtime).memories.add(text)
        return "Saved."

    @tool("recall")
    def recall(query: str, runtime: ToolRuntime[Any]) -> str:
        """Look up what is known about the user: their preferences, people, plans and what they said before,
        each line with where it came from. When nothing is known, it says so."""
        return sync_mem(runtime).context(query)

    @tool("remember")
    def remember(text: str, runtime: ToolRuntime[Any]) -> str:
        """Save something the user asked you to remember, in their words. Never passwords, keys or other
        secrets."""
        sync_mem(runtime).memories.add(text)
        return "Saved."

    recall.coroutine = arecall
    remember.coroutine = aremember
    return [recall, remember]
