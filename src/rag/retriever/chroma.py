from typing import TYPE_CHECKING

from langchain_community.vectorstores import Chroma

from src.rag.abstract.retriever import AbstractRetriever
from src.config import BASE_DIR

if TYPE_CHECKING:
    from langchain_core.embeddings.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStoreRetriever


class ChromaRetriever(AbstractRetriever):
    def __init__(
            self,
            embeddings_model: "Embeddings",
            path: str = str(BASE_DIR / "chroma"),
    ) -> None:
        self._chroma_db = Chroma(
            persist_directory=path,
            embedding_function=embeddings_model
        )

    def get_embeddings_retriever(self, k: int = 5) -> "VectorStoreRetriever":
        embedding_retriever = self._chroma_db.as_retriever(
            search_kwargs={
                "k": k
            }
        )
        return embedding_retriever
