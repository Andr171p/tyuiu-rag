from typing import Dict

from langchain_core.embeddings.embeddings import Embeddings
from langchain.embeddings import HuggingFaceEmbeddings

from src.config import settings


class EmbeddingsModel:
    def __init__(
            self,
            model_name: str = settings.embeddings.model_name,
            model_kwargs: Dict[str, str] = settings.embeddings.model_kwargs,
            encode_kwargs: Dict[str, bool] = settings.embeddings.encode_kwargs
    ) -> None:
        self._model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    def __call__(self, *args, **kwargs) -> Embeddings:
        return self._model
