from typing import List

from langchain_core.vectorstores.base import BaseRetriever
from langchain.retrievers import EnsembleRetriever


class RetrieverFactory:
    def __init__(
            self,
            retrievers: List[BaseRetriever],
            weights: List[float]
    ) -> None:
        if len(retrievers) != len(weights):
            raise ValueError("Count of retrievers and count weights must be same!")
        self._retrievers = retrievers
        self._weights = weights

    def create_retriever(self) -> BaseRetriever:
        return EnsembleRetriever(
            retrievers=self._retrievers,
            weights=self._weights
        )
