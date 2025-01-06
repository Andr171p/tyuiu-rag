from typing import TYPE_CHECKING, Dict

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings

if TYPE_CHECKING:
    from langchain_core.embeddings.embeddings import Embeddings


class EmbeddingsModel:
    def __init__(
            self,
            model_name: str = settings.embeddings.model_name,
            model_kwargs: Dict[str, str] = settings.embeddings.model_kwargs,
            encode_kwargs: Dict[str, bool] = settings.embeddings.encode_kwargs
    ) -> None:
        self._embeddings_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    def embeddings_function(self) -> "Embeddings":
        return self._embeddings_model
