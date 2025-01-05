from typing import List

from langchain_core.documents.base import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import Chroma

from src.rag.embeddings.model import EmbeddingsModel
from src.config import BASE_DIR


class ChromaDB:
    def __init__(
            self,
            path: str = str(BASE_DIR / "chroma"),
    ) -> None:
        self._chroma_db = Chroma(
            persist_directory=path,
            embedding_function=EmbeddingsModel()()
        )

    async def search_documents(
            self, query: str,
            k: int = 4
    ) -> List[Document]:
        documents = await self._chroma_db.asimilarity_search(
            query=query,
            k=k
        )
        return documents

    def get_embedding_retriever(self, k: int = 5) -> VectorStoreRetriever:
        embedding_retriever = self._chroma_db.as_retriever(
            search_kwargs={
                "k": k
            }
        )
        return embedding_retriever


c = ChromaDB()
import asyncio
print(asyncio.run(c.search_documents("Строин")))
