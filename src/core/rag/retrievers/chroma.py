from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_community.vectorstores import Chroma
    from langchain_core.vectorstores import VectorStoreRetriever


class ChromaRetrieverFactory:
    def __init__(
            self,
            chroma: "Chroma",
            k: int = 5
    ) -> None:
        self._chroma = chroma
        self._k = k

    def create_retriever(self) -> "VectorStoreRetriever":
        return self._chroma.as_retriever(
            search_kwargs={
                "k": self._k
            }
        )
