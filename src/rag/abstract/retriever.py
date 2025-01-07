from typing import TYPE_CHECKING

from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStoreRetriever


class AbstractRetriever(ABC):
    @abstractmethod
    def get_embeddings_retriever(self, k: int = 5) -> "VectorStoreRetriever":
        raise NotImplementedError("get_embeddings_retriever method is not implemented")
