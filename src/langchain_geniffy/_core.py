"""What the middleware, the tools and the retriever share: whose memory a call is for, and the words of a
message whichever shape LangChain hands it in."""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Union

Space = Union[str, None, Callable[[Any], Optional[str]]]
Named = Union[str, None, Callable[[Any], Optional[str]]]
NAME = "langchain-geniffy"

# What Geniffy keeps of a message, so nothing larger is sent: a turn's words, a tool call's input and a tool's output.
TURN_CHARS = 50_000
CALL_CHARS = 2_000
RESULT_CHARS = 4_000

_NO_SPACE = (
    "Geniffy needs a space: the user this is for, such as space=lambda runtime: f\"user_{runtime.context.id}\". "
    "Pass space=None only for your own memory, never for your users' data."
)


def check_space(space: Any) -> Optional[str]:
    """A space as given: None for your own memory, or a name. A blank name is refused rather than read as
    your own memory, because it is what an empty user id looks like, and that user would land in it."""
    if space is None:
        return None
    if not isinstance(space, str):
        raise TypeError(f"space must be a string (or None for your own memory), not {type(space).__name__}")
    if not space.strip():
        raise ValueError("space is blank: it would be your own memory, not a user's. " + _NO_SPACE)
    return space


def require_space(space: Any) -> None:
    if space is ...:
        raise TypeError(_NO_SPACE)
    if not callable(space):
        check_space(space)


def space_for(space: Space, runtime: Any) -> Optional[str]:
    """The space a call is for: the string, or what the function says for this runtime. A function names a
    user every time: None from it is refused, as a blank name is, since it is what a missing user id looks
    like. For your own memory, pass space=None itself."""
    if callable(space):
        named = space(runtime)
        if named is None:
            raise ValueError("space returned None for this call, so there is no user to read or save for. "
                             "For your own memory, pass space=None itself rather than a function.")
        if not isinstance(named, str):
            raise TypeError(f"space must return a string, not {type(named).__name__}")
        return check_space(named)
    return space


def bound(client: Any, space: Optional[str]) -> Any:
    """The client, pointed at one user's space (or your own memory, for None)."""
    return client if space is None else client.space(space)


def text_of(message: Any) -> str:
    """The words of a message: langchain-core's .text, or content as a string or a list of blocks."""
    text = getattr(message, "text", None)
    if isinstance(text, str):               # langchain-core 1.x: a property (a str that can also be called)
        return str(text).strip()
    if callable(text):                      # before 1.0: a method
        text = text()
        if isinstance(text, str):
            return text.strip()
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content
                         if not isinstance(b, dict) or b.get("type") == "text").strip()
    return ""


def last_of(messages: Any, kind: str) -> str:
    """The words of the last message of a kind ("human" or "ai")."""
    for message in reversed(list(messages or [])):
        if getattr(message, "type", None) == kind:
            return text_of(message)
    return ""


def named(value: Named, runtime: Any, what: str) -> Optional[str]:
    """A project or a session as given: a string, None, or what a function of the runtime says for this call."""
    if callable(value):
        value = value(runtime)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a string, not {type(value).__name__}")
    return value.strip() or None


def turn_start(messages: List[Any]) -> int:
    """Where the turn being answered starts: just after the last answer before the user's last message, so two
    messages sent at once are both in it."""
    human = [i for i, m in enumerate(messages) if getattr(m, "type", None) == "human"]
    if not human:
        return 0
    for i in range(human[-1] - 1, -1, -1):
        if getattr(messages[i], "type", None) == "ai":
            return i + 1
    return 0


def title_of(messages: List[Any]) -> Optional[str]:
    """What to call a saved conversation in the Geniffy app: the start of its first question, on one line."""
    for m in messages:
        if getattr(m, "type", None) == "human":
            words = " ".join(text_of(m).split())
            if words:
                return words if len(words) <= 80 else words[:80].rsplit(" ", 1)[0] + "..."
    return None


def _call(call: Any) -> Dict[str, Any]:
    args = call.get("args")
    try:
        shown = json.dumps(args, default=str)
    except (TypeError, ValueError):
        shown = str(args)
    out: Dict[str, Any] = {"name": str(call.get("name") or "tool")[:200],
                           "args": args if len(shown) <= CALL_CHARS else shown[:CALL_CHARS]}
    if call.get("id"):
        out["id"] = str(call["id"])[:200]
    return out


def as_sent(message: Any) -> Optional[Dict[str, Any]]:
    """A LangChain message as Geniffy reads a conversation: what the user said, what the agent answered with the
    tools it called, and what each tool returned. System messages are the app's instructions, not something anyone
    said, and are left out."""
    kind = getattr(message, "type", None)
    if kind == "human":
        out: Dict[str, Any] = {"role": "user", "content": text_of(message)[:TURN_CHARS]}
        if getattr(message, "name", None):
            out["name"] = str(message.name)[:200]
        return out
    if kind == "ai":
        out = {"role": "assistant", "content": text_of(message)[:TURN_CHARS]}
        calls = [_call(c) for c in (getattr(message, "tool_calls", None) or [])[:100] if isinstance(c, dict)]
        if calls:
            out["tool_calls"] = calls
        return out
    if kind == "tool":
        out = {"role": "tool", "content": text_of(message)[:RESULT_CHARS]}
        if getattr(message, "name", None):
            out["name"] = str(message.name)[:200]
        if getattr(message, "tool_call_id", None):
            out["tool_call_id"] = str(message.tool_call_id)[:200]
        return out
    return None


def said(sent: Dict[str, Any]) -> bool:
    """Whether a message as sent holds anything: words, a tool call or a tool's output."""
    return bool(sent.get("content") or sent.get("tool_calls") or sent.get("role") == "tool")


def made(cls: Any) -> Any:
    """A client of the geniffy SDK, named after this package, so the Requests page in the Geniffy app shows
    which integration made each call. A geniffy before 0.2.0 has no name to give, and makes a plain client."""
    from . import __version__
    try:
        return cls(integration=f"{NAME}/{__version__}")
    except TypeError:
        return cls()
