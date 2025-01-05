from typing import TYPE_CHECKING

from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from src.rag.generator.abstract.credentials import AbstractCredentials
    from src.rag.generator.abstract.llm import AbstractModel


class AbstractFactory(ABC):
    @staticmethod
    @abstractmethod
    def create_llm(self, credentials: AbstractCredentials) -> AbstractModel: pass
