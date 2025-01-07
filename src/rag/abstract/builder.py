from typing import (
    TYPE_CHECKING,
    Self
)

from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from src.rag.abstract.auth import AbstractAuth
    from src.rag.abstract.chain import AbstractChain
    from src.rag.abstract.embeddings import AbstractEmbeddingsFactory


class AbstractBuilder(ABC):
    @abstractmethod
    def set_retriever(self, embeddings_factory: "AbstractEmbeddingsFactory") -> Self:
        raise NotImplementedError("set_retriever method is not implemented")

    @abstractmethod
    def set_llm_chain(self, auth: "AbstractAuth", template: str) -> Self:
        raise NotImplementedError("set_llm_chain method is not implemented")

    @abstractmethod
    def get_rag_chain(self) -> "AbstractChain":
        raise NotImplementedError("get_rag_chain method is not implemented")
