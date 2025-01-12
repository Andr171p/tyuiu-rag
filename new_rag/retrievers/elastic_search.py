from langchain.retrievers import ElasticSearchBM25Retriever

from src.config import settings


class ElasticSearchRetrieverFactory:
    def __init__(
            self,
            url: str = settings.es.url,
            index_name: str = settings.es.index_name
    ) -> None:
        self._url = url
        self._index_name = index_name

    def create_retriever(self) -> ElasticSearchBM25Retriever:
        return ElasticSearchBM25Retriever.create(
            elasticsearch_url=self._url,
            index_name=self._index_name
        )
