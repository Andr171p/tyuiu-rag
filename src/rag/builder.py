from typing import (
    TYPE_CHECKING,
    Optional
)

from src.rag.abstract import (
    AbstractEmbeddingsFactory,
    AbstractBuilder,
    AbstractAuth,
    AbstractChain
)
from src.rag.retriever.chroma import ChromaRetriever
from src.rag.generator.llm import GigaChatLLM
from src.rag.chain import RAGChain

if TYPE_CHECKING:
    from langchain_core.runnables.base import Runnable
    from langchain_core.vectorstores import VectorStoreRetriever


class RAGBuilder(AbstractBuilder):
    __slots__ = ("_retriever", "_llm_chain")

    def __init__(self) -> None:
        self._retriever: Optional["VectorStoreRetriever"] = None
        self._llm_chain: Optional["Runnable"] = None

    def set_retriever(self, embeddings_factory: AbstractEmbeddingsFactory) -> "RAGBuilder":
        embeddings_model = embeddings_factory.create_embeddings_model()
        chroma = ChromaRetriever(embeddings_model)
        self._retriever = chroma.get_embeddings_retriever()
        return self

    def set_llm_chain(self,  auth: AbstractAuth, template: str) -> "RAGBuilder":
        llm = GigaChatLLM(auth)
        self._llm_chain = llm.create_llm_chain(template)
        return self

    def get_rag_chain(self) -> AbstractChain | None:
        if not all(getattr(self, attr) for attr in self.__slots__):
            raise ValueError("not all attributes are set")
        return RAGChain(
            retriever=self._retriever,
            llm_chain=self._llm_chain
        )


'''from src.rag.retriever.embeddings import EmbeddingsFactory
from src.rag.generator.auth import GigaChatAuth
from src.misc.file import load_txt
from src.config import settings
async def main() -> None:
    print(settings.giga_chat.prompt)
    template = await load_txt(r"C:\Users\andre\TyuiuRAG\static\prompt\chat.txt")
    rag_builder = (
        RAGBuilder()
        .set_retriever(EmbeddingsFactory())
        .set_llm_chain(GigaChatAuth(), template)
    )
    rag_chain = rag_builder.get_rag_chain()
    print(await rag_chain.invoke("Расскажи о ТИУ"))


import asyncio
asyncio.run(main())'''
