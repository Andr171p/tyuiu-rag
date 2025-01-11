from chromadb import HttpClient
from chromadb.config import Settings


class ChromaVectorStore:
    def __init__(
            self,
            host: str,
            port: int,
            settings: Settings
    ) -> None:
        self._chroma_db = HttpClient(
            host=host,
            port=port,
            settings=settings
        )

    def add_documents(self, documents, embeddings) -> None:
        self._chroma_db.fro