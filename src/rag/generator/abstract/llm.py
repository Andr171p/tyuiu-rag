from abc import ABC, abstractmethod

from langchain_core.language_models.chat_models import BaseChatModel


class AbstractModel(ABC):
    @abstractmethod
    async def predict(self, text: str) -> str: pass
