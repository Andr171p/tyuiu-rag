from typing import List

from langchain_core.documents.base import Document
from langchain_community.vectorstores import Chroma
from chromadb import PersistentClient

from src.config import BASE_DIR


from langchain.embeddings import HuggingFaceEmbeddings

from src.embeddings.function import get_embeddings, ChromaEmbeddingFunction


class ChromaDB:
    def __init__(
            self,
            path: str = str(BASE_DIR / "chromadb_data"),
    ) -> None:
        self._chroma_db = Chroma(
            persist_directory=path,
            embedding_function=get_embeddings()
        )

    def similarity_search(
            self, query: str,
            k: int = 4
    ) -> List[Document]:
        documents = self._chroma_db.similarity_search(
            query=query,
            k=k
        )
        return documents


c = ChromaDB()
'''model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding = HuggingFaceEmbeddings(model_name=model_name,
                                  model_kwargs=model_kwargs,
                                  encode_kwargs=encode_kwargs)'''
print(c.similarity_search("Строин"))
