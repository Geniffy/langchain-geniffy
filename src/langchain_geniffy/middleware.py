"""GeniffyMemory: a memory of each user around any LangChain agent, as one piece of middleware."""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional, Tuple

from langchain.agents.middleware import AgentMiddleware

from ._core import Space, bound, last_of, require_space, space_for

logger = logging.getLogger("langchain_geniffy")

DEFAULT_INSTRUCTIONS = (
    "What follows is this user's memory: what they told this app before, each line with where it came from. "
    "Use it when it helps and don't recite it. If it doesn't cover something, say so instead of guessing."
)


class _Recent:
    """What was just looked up, for a few seconds: an agent calls the model once per step, with the same
    question, and Geniffy is asked once. Keyed by space and question, so no user ever sees another's."""

    def __init__(self, seconds: float = 30.0, most: int = 512):
        self.seconds, self.most = seconds, most
        self._held: "OrderedDict[Tuple[Optional[str], str], Tuple[float, str]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Tuple[Optional[str], str]) -> Optional[str]:
        with self._lock:
            hit = self._held.get(key)
            if hit and time.monotonic() - hit[0] < self.seconds:
                return hit[1]
            self._held.pop(key, None)
            return None

    def put(self, key: Tuple[Optional[str], str], context: str) -> None:
        with self._lock:
            self._held[key] = (time.monotonic(), context)
            while len(self._held) > self.most:
                self._held.popitem(last=False)


class GeniffyMemory(AgentMiddleware):
    """Before each call to the model, what Geniffy knows about the user that bears on their last message
    goes in the system prompt, after your own. When the agent has its answer, the exchange is saved.

        agent = create_agent(
            "anthropic:claude-opus-5-5",
            middleware=[GeniffyMemory(space=lambda runtime: f"user_{runtime.context.id}")],
            context_schema=User,
        )

    space        the user: a string, or a function of the runtime (required; None is your own memory)
    remember     save each finished exchange (default True)
    instructions what the model is told about the memory it is given
    client       a geniffy.Geniffy for sync agents (default: one that reads GENIFFY_API_KEY)
    async_client a geniffy.AsyncGeniffy for async agents (default: likewise)
    on_error     called when Geniffy can't be reached; the agent goes on without memory
    """

    def __init__(self, *, space: Space = ..., remember: bool = True, instructions: str = DEFAULT_INSTRUCTIONS,
                 client: Any = None, async_client: Any = None,
                 on_error: Optional[Callable[[Exception], None]] = None):
        super().__init__()
        require_space(space)
        self.space = space
        self.remember = remember
        self.instructions = instructions
        self._client, self._async_client = client, async_client
        self._on_error = on_error or (lambda e: logger.warning("Geniffy: %s", e))
        self._recent = _Recent()

    # the clients, made when first needed, so importing never needs a key
    def _sync(self) -> Any:
        if self._client is None:
            from geniffy import Geniffy
            self._client = Geniffy()
        return self._client

    def _async(self) -> Any:
        if self._async_client is None:
            from geniffy import AsyncGeniffy
            self._async_client = AsyncGeniffy()
        return self._async_client

    def _with_memory(self, request: Any, context: Optional[str]) -> Any:
        if not context:
            return request
        block = f"{self.instructions}\n\n<memory>\n{context}\n</memory>"
        system = f"{request.system_prompt}\n\n{block}" if request.system_prompt else block
        return request.override(system_prompt=system)

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        question = last_of(request.messages, "human")
        context = None
        if question:
            space = space_for(self.space, request.runtime)
            context = self._recent.get((space, question))
            if context is None:
                try:
                    context = bound(self._sync(), space).context(question)
                    self._recent.put((space, question), context)
                except Exception as e:  # noqa: BLE001 - the agent goes on without memory
                    self._on_error(e)
        return handler(self._with_memory(request, context))

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        question = last_of(request.messages, "human")
        context = None
        if question:
            space = space_for(self.space, request.runtime)
            context = self._recent.get((space, question))
            if context is None:
                try:
                    context = await bound(self._async(), space).context(question)
                    self._recent.put((space, question), context)
                except Exception as e:  # noqa: BLE001
                    self._on_error(e)
        return await handler(self._with_memory(request, context))

    def _exchange(self, state: Any) -> Optional[list]:
        messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
        asked, answer = last_of(messages, "human"), last_of(messages, "ai")
        if not self.remember or not asked or not answer:
            return None
        return [{"role": "user", "content": asked}, {"role": "assistant", "content": answer}]

    def after_agent(self, state: Any, runtime: Any) -> None:
        exchange = self._exchange(state)
        if exchange:
            try:
                bound(self._sync(), space_for(self.space, runtime)).memories.add(messages=exchange)
            except Exception as e:  # noqa: BLE001
                self._on_error(e)
        return None

    async def aafter_agent(self, state: Any, runtime: Any) -> None:
        exchange = self._exchange(state)
        if exchange:
            try:
                await bound(self._async(), space_for(self.space, runtime)).memories.add(messages=exchange)
            except Exception as e:  # noqa: BLE001
                self._on_error(e)
        return None
