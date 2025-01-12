from src.core.rag import ChainBuilder
from src.core.rag.embeddings import EmbeddingsModel
from src.core.rag.vector_store import ChromaVectorStore
from src.core.rag.llm import GigaChatLLM
from src.core.rag.retrievers import (
    ChromaRetrieverFactory,
    ElasticSearchRetrieverFactory
)


class ChatService:
    def __init__(
            self,
            embeddings: EmbeddingsModel = EmbeddingsModel(),
            llm: GigaChatLLM = GigaChatLLM()
    ) -> None:
        self._chroma_retriever_factory = ChromaRetrieverFactory(ChromaVectorStore(embeddings))
        self._es_retriever_factory = ElasticSearchRetrieverFactory()

