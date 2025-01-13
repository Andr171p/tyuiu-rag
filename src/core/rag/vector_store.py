from typing import TYPE_CHECKING, Any

from chromadb import AsyncHttpClient
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


class ChromaVectorStore(Chroma):
    def __init__(
            self,
            embeddings: "Embeddings",
            client: Any = AsyncHttpClient(),
            settings: Settings = Settings()
    ) -> None:
        super().__init__(
            client=client,
            client_settings=settings,
            embedding_function=embeddings
        )
