from typing import (
    TYPE_CHECKING,
    List
)

from langchain.retrievers import EnsembleRetriever

if TYPE_CHECKING:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.vectorstores import VectorStore


class EnsembleRetrieverBuilder:
    _retrievers: List["BaseRetriever"] = None
    _weights: List[float] = None

    @classmethod
    def set_documents_retriever(
            cls,
            retriever: "BaseRetriever",
            weight: float = 0.4
    ) -> "EnsembleRetrieverBuilder":
        cls._retrievers.append(retriever)
        cls._weights.append(weight)
        return cls()

    @classmethod
    def set_vector_store_retriever(
            cls,
            vector_store: "VectorStore",
            k: int = 5,
            weight: float = 0.6
    ) -> "EnsembleRetrieverBuilder":
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        cls._retrievers.append(retriever)
        cls._weights.append(weight)
        return cls()

    @classmethod
    def create_ensemble_retriever(cls) -> "BaseRetriever":
        if len(cls._retrievers) != len(cls._weights):
            raise ValueError("Count retrievers and count weights must be same")
        return EnsembleRetriever(
            retrievers=cls._retrievers,
            weights=cls._weights
        )
