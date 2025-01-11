from langchain.embeddings import HuggingFaceEmbeddings

from src.config import settings


class E5BaseEmbeddingsModel(HuggingFaceEmbeddings):
    model_config = {
        "model_name": settings.embeddings.model_name,
        "model_kwargs": settings.embeddings.model_kwargs,
        "encode_kwargs": settings.embeddings.encode_kwargs
    }
