"""GeniffyRetriever: the memories that best match a query, as LangChain Documents."""
from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, field_validator

from ._core import check_space


def _document(memory: Any) -> Document:
    source = getattr(memory, "source", None)
    return Document(page_content=memory.text, metadata={
        "id": memory.id, "kind": memory.kind, "about": memory.about, "said_at": memory.said_at,
        "learned_at": memory.learned_at, "source": getattr(source, "title", None),
    })


class GeniffyRetriever(BaseRetriever):
    """The memories that best match a query, best first, each a Document with when it was said and where it
    came from. Ranked, not judged: for "what is known that bears on this", use GeniffyMemory or context().

        retriever = GeniffyRetriever(space="user_1042", k=5)
        docs = retriever.invoke("the Lumen renewal")

    space is required: the user whose memory this reads (None is your own memory; a blank one is refused).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    space: Optional[str]
    k: int = 10
    client: Any = None
    async_client: Any = None

    @field_validator("space", mode="before")
    @classmethod
    def _one_user(cls, space: Any) -> Optional[str]:
        return check_space(space)

    def _mem(self) -> Any:
        if self.client is None:
            from geniffy import Geniffy
            self.client = Geniffy()
        return self.client if self.space is None else self.client.space(self.space)

    def _amem(self) -> Any:
        if self.async_client is None:
            from geniffy import AsyncGeniffy
            self.async_client = AsyncGeniffy()
        return self.async_client if self.space is None else self.async_client.space(self.space)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return [_document(m) for m in self._mem().search(query, limit=self.k)]

    async def _aget_relevant_documents(self, query: str, *,
                                       run_manager: AsyncCallbackManagerForRetrieverRun) -> List[Document]:
        return [_document(m) for m in await self._amem().search(query, limit=self.k)]
