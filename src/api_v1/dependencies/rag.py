from src.api_v1.container import rag_builder
from src.rag.abstract import AbstractChain


def get_rag_chain() -> AbstractChain:
    rag_chain: AbstractChain = rag_builder.get_rag_chain()
    return rag_chain
