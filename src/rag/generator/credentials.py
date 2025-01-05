import base64

from src.rag.generator.abstract.credentials import AbstractCredentials
from src.config import settings


class GigaChatCredentials(AbstractCredentials):
    def __init__(
            self,
            client_id: str = settings.giga_chat.client_id,
            client_secret: str = settings.giga_chat.client_secret,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret

    def get_auth_key(self) -> str:
        credentials = f"{self._client_id}:{self._client_secret}"
        return base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
