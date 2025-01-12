from typing import TYPE_CHECKING, List, Optional

from langchain.retrievers import EnsembleRetriever

if TYPE_CHECKING:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.prompts.chat import BaseChatPromptTemplate
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.runnables.base import Runnable


class ChainBuilder:
    __slots__ = ("_ensemble_retriever", "_chat_prompt", "_llm")

    def __init__(self) -> None:
        self._ensemble_retriever: Optional["BaseRetriever"] = None
        self._chat_prompt: Optional["BaseChatPromptTemplate"] = None
        self._llm: Optional["BaseChatModel"] = None

    def set_ensemble_retriever(
            self,
            retrievers: List[BaseRetriever],
            weights: List[float]
    ) -> "ChainBuilder":
        if len(retrievers) != len(weights):
            raise ValueError("Count of retrievers and count of weights must be same")
        self._ensemble_retriever = EnsembleRetriever(
            retrievers=retrievers,
            weights=weights
        )
        return self

    def set_chat_prompt(
            self,
            chat_prompt: BaseChatPromptTemplate
    ) -> "ChainBuilder":
        self._chat_prompt = chat_prompt
        return self

    def create_chain(self) -> "Runnable":
        if not all(getattr(self, attr) for attr in self.__slots__):
            raise ValueError("Not all attributes are set")
        chain = (
                self._ensemble_retriever |
                self._chat_prompt |
                self._llm
        )
        return chain
