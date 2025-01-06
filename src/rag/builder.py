from typing import Optional

from src.rag.abstract import (
    AbstractVectorStore,
    AbstractAuth,
    AbstractModel,
    AbstractChain
)
from src.rag.chain import RAGChain

from langchain_core.runnables.base import Runnable
from langchain_core.vectorstores import VectorStoreRetriever


class RAGBuilder:
    __slots__ = ("_embedding_retriever", "_documents_chain")

    def __init__(self) -> None:
        self._embedding_retriever: Optional[VectorStoreRetriever] = None
        self._documents_chain: Optional[Runnable] = None

    def set_vector_store(self, vector_store: AbstractVectorStore) -> None:
        self._embedding_retriever = vector_store.get_embeddings_retriever()

    def set_llm(self,  model: AbstractModel, template: str) -> None:
        self._documents_chain = model.create_documents_chain(template)

    def get_rag_chain(self) -> AbstractChain | None:
        if not all(getattr(self, attr) for attr in self.__slots__):
            raise ValueError("not all attributes are set")
        return RAGChain(
            retriever=self._embedding_retriever,
            documents_chain=self._documents_chain
        )
