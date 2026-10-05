"""What the middleware, the tools and the retriever share: whose memory a call is for, and the words of a
message whichever shape LangChain hands it in."""
from __future__ import annotations

from typing import Any, Callable, Optional, Union

Space = Union[str, None, Callable[[Any], Optional[str]]]

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
