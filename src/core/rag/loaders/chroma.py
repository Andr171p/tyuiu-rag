from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings


class ChromaLoader:
    def __init__(self, chroma: "Chroma") -> None:
        self._chroma = chroma

    def load_documents(
            self,
            documents: List["Document"],
            embeddings: "Embeddings"
    ) -> None:
        self._chroma.from_documents(documents, embeddings)
