from typing import Any, List, Dict, AsyncIterator

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from langchain_core.documents import Document

from src.config import settings


class ElasticSearchLoader:
    def __init__(
            self,
            url: str = settings.es.url
    ) -> None:
        self._client = AsyncElasticsearch(url)

    @staticmethod
    async def _generate_documents(
            documents: List[Document]
    ) -> AsyncIterator[Dict[str, Any]]:
        for i, document in enumerate(documents):
            doc = {
                "_index": "tyuiu",
                "doc": {
                    "id": i,
                    "content": document.page_content
                }
            }
            yield doc

    async def load_documents(
            self,
            documents: List[Document]
    ) -> None:
        await async_bulk(self._client, self._generate_documents(documents))
