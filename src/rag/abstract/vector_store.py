from abc import ABC, abstractmethod

from langchain_core.vectorstores import VectorStoreRetriever


class AbstractVectorStore(ABC):
    @abstractmethod
    def get_embeddings_retriever(self, k: int = 5) -> VectorStoreRetriever:
        raise NotImplementedError("get_embeddings_retriever method is not implemented")
