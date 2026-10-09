"""Geniffy for LangChain: a memory of each of your users, for any LangChain agent.

    GeniffyMemory(space=...)    middleware: the user's briefing goes in the system prompt before each model
                                call, and each run is saved, tool calls included, into one memory for the
                                whole conversation
    geniffy_tools(space=...)    recall and remember, as tools the agent calls itself
    GeniffyRetriever(space=...) the memories that match a query, as Documents, for a chain

`space` names the user: a string, or a function of the agent's runtime, such as
`lambda runtime: f"user_{runtime.context.id}"`, so one agent serves every user. It is required, so one user's
memory is never written to another's; `space=None` is your own memory, never your users' data.
"""
from ._core import Space
from .middleware import GeniffyMemory
from .retriever import GeniffyRetriever
from .tools import geniffy_tools

__all__ = ["GeniffyMemory", "GeniffyRetriever", "Space", "geniffy_tools"]
__version__ = "0.2.0"
