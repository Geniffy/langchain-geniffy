"""GeniffyMemory: a memory of each user around any LangChain agent, as one piece of middleware."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import PrivateStateAttr
from typing_extensions import Annotated, NotRequired

from ._core import (Named, Space, as_sent, bound, last_of, made, named, require_space, said, space_for, title_of,
                    turn_start)

logger = logging.getLogger("langchain_geniffy")

DEFAULT_INSTRUCTIONS = (
    "What follows is this user's memory: what this app knows from before, each line with its date. "
    "Use it when it helps and don't recite it. If it doesn't cover something, say so instead of guessing."
)
# A briefing with nothing in it yet still says so: a model reads an empty memory as permission to invent.
NOTHING_YET = "Nothing is remembered about this user yet."
PER_CALL = 500          # the most messages Geniffy takes in one call
OFF_SECONDS = 600.0     # how long the memories alone are read after the briefing was found switched off
START = "(start)"       # the mark before a conversation's first message: none of it is saved yet


class SavedState(AgentState):
    """The agent's state, with where saving got to: the id of the last message before what is still to be sent.
    With a checkpointer it is kept with the thread, so a conversation is never sent twice, across restarts and on
    every server, and a save that failed goes with the next run. It is the middleware's own: never in what the
    agent takes in or hands back."""

    geniffy_saved: NotRequired[Annotated[str, PrivateStateAttr]]


class _Recent:
    """What was just looked up, for a few seconds: an agent calls the model once per step, with the same
    question, and Geniffy is asked once. Keyed by space, project and question, so no user ever sees another's."""

    def __init__(self, seconds: float = 30.0, most: int = 512):
        self.seconds, self.most = seconds, most
        self._held: "OrderedDict[Tuple[Optional[str], Optional[str], str], Tuple[float, str]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Tuple[Optional[str], Optional[str], str]) -> Optional[str]:
        with self._lock:
            hit = self._held.get(key)
            if hit and time.monotonic() - hit[0] < self.seconds:
                return hit[1]
            self._held.pop(key, None)
            return None

    def put(self, key: Tuple[Optional[str], Optional[str], str], context: str) -> None:
        with self._lock:
            self._held[key] = (time.monotonic(), context)
            while len(self._held) > self.most:
                self._held.popitem(last=False)


def _thread_of(runtime: Any) -> Optional[str]:
    """The conversation a run belongs to: its LangGraph thread, when the agent has a checkpointer."""
    thread = getattr(getattr(runtime, "execution_info", None), "thread_id", None)
    if thread is None:
        try:
            from langgraph.config import get_config
            thread = (get_config().get("configurable") or {}).get("thread_id")
        except Exception:  # noqa: BLE001 - outside a run, or async on Python 3.10 with an older LangGraph
            thread = None
    return str(thread) if thread not in (None, "") else None


def _state(state: Any, key: str) -> Any:
    return state.get(key) if isinstance(state, dict) else getattr(state, key, None)


class GeniffyMemory(AgentMiddleware):
    """Before each call to the model, the user's briefing goes in the system prompt, after your own: where things
    stand, what is due, the rules that apply, what happened, and what is known that bears on their last message.
    When the agent has its answer, the run is saved, tool calls and results included, into one memory for the
    whole conversation.

        agent = create_agent(
            "anthropic:claude-opus-5-5",
            middleware=[GeniffyMemory(space=lambda runtime: f"user_{runtime.context.id}")],
            context_schema=User,
            checkpointer=InMemorySaver(),
        )

    space        the user: a string, or a function of the runtime that names one every time (required;
                 space=None is your own memory, and a blank space is refused)
    project      the project the agent works on: a string, or a function of the runtime. The briefing keeps to
                 it, and what is saved is labelled with it. Left out: every project
    session      the conversation: a string, or a function of the runtime. Left out: the run's LangGraph thread,
                 when the agent has a checkpointer; with neither, each run is a memory of its own
    briefing     open with the briefing (default True); False reads only what is known that bears on the
                 user's last message. An account without the briefing reads that, either way
    budget_chars the most the briefing adds to the prompt, in characters (500 to 40,000; default 6,000)
    remember     save each run (default True)
    instructions what the model is told about the memory it is given
    client       a geniffy.Geniffy for sync agents (default: one that reads GENIFFY_API_KEY)
    async_client a geniffy.AsyncGeniffy for async agents (default: likewise)
    on_error     called when Geniffy can't be reached; the agent goes on without memory
    """

    state_schema = SavedState

    def __init__(self, *, space: Space = ..., project: Named = None, session: Named = None, briefing: bool = True,
                 budget_chars: int = 6000, remember: bool = True, instructions: str = DEFAULT_INSTRUCTIONS,
                 client: Any = None, async_client: Any = None,
                 on_error: Optional[Callable[[Exception], None]] = None):
        super().__init__()
        require_space(space)
        self.space, self.project, self.session = space, project, session
        self.briefing, self.budget_chars = briefing, budget_chars
        self.remember = remember
        self.instructions = instructions
        self._client, self._async_client = client, async_client
        self._on_error = on_error or (lambda e: logger.warning("Geniffy: %s", e))
        self._recent = _Recent()
        self._off_until = 0.0

    # the clients, made when first needed, so importing never needs a key
    def _sync(self) -> Any:
        if self._client is None:
            from geniffy import Geniffy
            self._client = made(Geniffy)
        return self._client

    def _async(self) -> Any:
        if self._async_client is None:
            from geniffy import AsyncGeniffy
            self._async_client = made(AsyncGeniffy)
        return self._async_client

    # ── before each model call: the briefing ────────────────────────────────────────────────────────────────
    def _briefing_on(self) -> bool:
        return self.briefing and time.monotonic() >= self._off_until

    def _switched_off(self, e: Exception) -> bool:
        """The briefing is not switched on for this key (404), or the API is from before it (405): what is known
        that bears on the question is read instead, and the briefing is tried again in ten minutes."""
        if getattr(e, "status", None) in (404, 405):
            self._off_until = time.monotonic() + OFF_SECONDS
            return True
        return False

    def _read(self, mem: Any, project: Optional[str], question: str) -> Optional[str]:
        if self._briefing_on():
            try:
                return mem.briefing(project=project, cue=question, budget_chars=self.budget_chars) or NOTHING_YET
            except Exception as e:  # noqa: BLE001
                if not self._switched_off(e):
                    raise
        return mem.context(question) if question else None

    async def _aread(self, mem: Any, project: Optional[str], question: str) -> Optional[str]:
        if self._briefing_on():
            try:
                return await mem.briefing(project=project, cue=question, budget_chars=self.budget_chars) or NOTHING_YET
            except Exception as e:  # noqa: BLE001
                if not self._switched_off(e):
                    raise
        return await mem.context(question) if question else None

    def _with_memory(self, request: Any, context: Optional[str]) -> Any:
        if not context:
            return request
        block = f"{self.instructions}\n\n<memory>\n{context}\n</memory>"
        system = f"{request.system_prompt}\n\n{block}" if request.system_prompt else block
        return request.override(system_prompt=system)

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        question = last_of(request.messages, "human")
        space = space_for(self.space, request.runtime)
        project = named(self.project, request.runtime, "project")
        context = self._recent.get((space, project, question))
        if context is None:
            try:
                context = self._read(bound(self._sync(), space), project, question)
                if context is not None:
                    self._recent.put((space, project, question), context)
            except Exception as e:  # noqa: BLE001 - the agent goes on without memory
                self._on_error(e)
        return handler(self._with_memory(request, context))

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        question = last_of(request.messages, "human")
        space = space_for(self.space, request.runtime)
        project = named(self.project, request.runtime, "project")
        context = self._recent.get((space, project, question))
        if context is None:
            try:
                context = await self._aread(bound(self._async(), space), project, question)
                if context is not None:
                    self._recent.put((space, project, question), context)
            except Exception as e:  # noqa: BLE001
                self._on_error(e)
        return await handler(self._with_memory(request, context))

    # ── after the run: the conversation, saved as it goes ──────────────────────────────────────────────────
    def _to_save(self, state: Any, runtime: Any) -> Optional[Dict[str, Any]]:
        """What a run adds to its conversation: the messages after the mark, or, the first time or when the marked
        one is gone (trimmed or summarized away), this turn, from what the user said on. `before` is the mark to
        keep if none of it can be sent, so it goes with the next run."""
        if not self.remember:
            return None
        messages: List[Any] = list(_state(state, "messages") or [])
        ids = [getattr(m, "id", None) for m in messages]
        mark = _state(state, "geniffy_saved")
        start = 0 if mark == START else (ids.index(mark) + 1 if mark and mark in ids else turn_start(messages))
        new = messages[start:]
        pairs = [(getattr(m, "id", None), s) for m, s in ((m, as_sent(m)) for m in new) if s is not None and said(s)]
        if not pairs:
            return None
        project = named(self.project, runtime, "project")
        return {"space": space_for(self.space, runtime), "pairs": pairs, "before": ids[start - 1] if start else START,
                "session": named(self.session, runtime, "session") or _thread_of(runtime) or f"run-{uuid.uuid4().hex}",
                "title": title_of(new) or title_of(messages), "labels": {"project": project} if project else None}

    @staticmethod
    def _mark(mark: Optional[str]) -> Optional[Dict[str, Any]]:
        return {"geniffy_saved": mark} if isinstance(mark, str) and mark else None

    def after_agent(self, state: Any, runtime: Any) -> Optional[Dict[str, Any]]:
        plan = self._to_save(state, runtime)
        if plan is None:
            return None
        pairs, last = plan["pairs"], None
        try:
            mem = bound(self._sync(), plan["space"])
            for i in range(0, len(pairs), PER_CALL):
                part = pairs[i:i + PER_CALL]
                mem.memories.add(messages=[s for _, s in part], session=plan["session"], title=plan["title"],
                                 labels=plan["labels"])
                last = part[-1][0]
        except Exception as e:  # noqa: BLE001 - what was not sent goes with the next run
            self._on_error(e)
        return self._mark(last or plan["before"])

    async def aafter_agent(self, state: Any, runtime: Any) -> Optional[Dict[str, Any]]:
        plan = self._to_save(state, runtime)
        if plan is None:
            return None
        pairs, last = plan["pairs"], None
        try:
            mem = bound(self._async(), plan["space"])
            for i in range(0, len(pairs), PER_CALL):
                part = pairs[i:i + PER_CALL]
                await mem.memories.add(messages=[s for _, s in part], session=plan["session"], title=plan["title"],
                                       labels=plan["labels"])
                last = part[-1][0]
        except Exception as e:  # noqa: BLE001
            self._on_error(e)
        return self._mark(last or plan["before"])
