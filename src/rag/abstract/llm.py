from abc import ABC, abstractmethod

from langchain_core.runnables.base import Runnable


class AbstractLLM(ABC):
    @abstractmethod
    def create_llm_chain(self, template: str) -> Runnable:
        raise NotImplementedError("create_documents_chain method is not implemented")
