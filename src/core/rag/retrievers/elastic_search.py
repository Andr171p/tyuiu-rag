from elasticsearch import Elasticsearch
from langchain.retrievers import ElasticSearchBM25Retriever, BM25Retriever

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
        return ElasticSearchBM25Retriever(
            client=Elasticsearch(self._url),
            index_name=self._index_name
        )


'''from elasticsearch import Elasticsearch

# Подключение к Elasticsearch
es = Elasticsearch(settings.es.url)

# Имя индекса, который нужно удалить
index_name = settings.es.index_name

# Удаление индекса
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)
    print(f"Индекс '{index_name}' успешно удалён.")
else:
    print(f"Индекс '{index_name}' не существует.")'''