from src.api_v1.app import app
from src.rag.abstract import AbstractChain, AbstractBuilder


def get_rag_chain() -> AbstractChain:
    rag_builder: AbstractBuilder = app.state.rag_builder
    rag_chain: AbstractChain = rag_builder.get_rag_chain()
    return rag_chain
