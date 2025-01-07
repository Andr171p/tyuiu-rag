from typing import (
    TYPE_CHECKING,
    Dict
)

from langchain_huggingface import HuggingFaceEmbeddings

from src.rag.abstract.embeddings import AbstractEmbeddingsFactory
from src.config import settings

if TYPE_CHECKING:
    from langchain_core.embeddings.embeddings import Embeddings


class EmbeddingsFactory(AbstractEmbeddingsFactory):
    def __init__(
            self,
            model_name: str = settings.embeddings.model_name,
            model_kwargs: Dict[str, str] = settings.embeddings.model_kwargs,
            encode_kwargs: Dict[str, bool] = settings.embeddings.encode_kwargs
    ) -> None:
        self._model_name = model_name
        self._model_kwargs = model_kwargs
        self._encode_kwargs = encode_kwargs

    def create_embeddings_model(self) -> "Embeddings":
        return HuggingFaceEmbeddings(
            model_name=self._model_name,
            model_kwargs=self._model_kwargs,
            encode_kwargs=self._encode_kwargs
        )
