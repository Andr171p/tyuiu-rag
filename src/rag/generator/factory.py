from src.rag.generator.abstract.credentials import AbstractCredentials
from src.rag.generator.abstract.llm import AbstractModel
from src.rag.generator.abstract.factory import AbstractFactory
from src.rag.generator.llm import GigaChatModel


class GigaChatFactory(AbstractFactory):
    @staticmethod
    def create_llm(self, credentials: AbstractCredentials) -> AbstractModel:
        auth_key = credentials.get_auth_key()
        return GigaChatModel(auth_key)
