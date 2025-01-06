from abc import ABC, abstractmethod

from langchain_core.runnables.base import Runnable


class AbstractChain(ABC):
    chain: Runnable

    @abstractmethod
    async def invoke(self, query: str) -> str:
        raise NotImplementedError("invoke method is not implemented")
