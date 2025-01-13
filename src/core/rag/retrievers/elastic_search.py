from elasticsearch import Elasticsearch
from langchain.retrievers import ElasticSearchBM25Retriever

from src.config import settings


class ElasticSearchRetriever(ElasticSearchBM25Retriever):
    def __init__(
            self,
            client: Elasticsearch = Elasticsearch(settings.es.url),
            index_name: str = settings.es.index_name
    ) -> None:
        super().__init__(
            client=client,
            index_name=index_name
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