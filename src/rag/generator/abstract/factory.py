from typing import TYPE_CHECKING

from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from src.rag.generator.abstract.llm import AbstractLLM


class AbstractFactory(ABC):
    @abstractmethod
    def create_llm(self, auth_key: str) -> AbstractLLM: pass
