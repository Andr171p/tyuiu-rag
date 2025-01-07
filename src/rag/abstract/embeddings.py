from typing import TYPE_CHECKING

from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from langchain_core.embeddings.embeddings import Embeddings


class AbstractEmbeddingsFactory(ABC):
    @abstractmethod
    def create_embeddings_model(self) -> "Embeddings":
        raise NotImplementedError("create_embeddings_model method is not implemented")
