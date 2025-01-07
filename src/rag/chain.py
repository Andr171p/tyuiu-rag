from typing import Dict

from langchain_core.runnables.base import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains.retrieval import create_retrieval_chain

from src.rag.abstract.chain import AbstractChain


class RAGChain(AbstractChain):
    def __init__(
            self,
            retriever: VectorStoreRetriever,
            llm_chain: Runnable
    ) -> None:
        self._chain: Runnable = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=llm_chain
        )

    async def invoke(self, query: str) -> str:
        input: Dict[str, str] = {"input": query}
        output = await self._chain.ainvoke(input)
        return output["answer"]
