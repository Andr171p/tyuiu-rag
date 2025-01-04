from chromadb import Documents, EmbeddingFunction, Embeddings

from langchain.embeddings import HuggingFaceEmbeddings

from src.config import settings


def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embeddings.model_name,
        model_kwargs=settings.embeddings.model_kwargs,
        encode_kwargs=settings.embeddings.encode_kwargs
    )
    return embeddings


class ChromaEmbeddingFunction(EmbeddingFunction):
    def __call__(self) -> Embeddings:
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embeddings.model_name,
            model_kwargs=settings.embeddings.model_kwargs,
            encode_kwargs=settings.embeddings.encode_kwargs
        )
        return embeddings
