import uuid
import base64
import logging

from src.http.client import HTTPClient
from src.config import settings


class GigaChatCredentials:
    def __init__(
            self,
            client_id: str = settings.giga_chat.client_id,
            client_secret: str = settings.giga_chat.client_secret,
            scope: str = settings.giga_chat.scope,
            url: str = settings.giga_chat.url
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._scope = scope
        self._url = url

    def get_auth_key(self) -> str:
        credentials = f"{self._client_id}:{self._client_secret}"
        return base64.b64encode(credentials.encode("utf-8")).decode("utf-8")

    async def get_token(self, http_client: HTTPClient) -> ...:
        import requests
        rq_uid = str(uuid.uuid4())
        auth_key: str = self.get_auth_key()
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": rq_uid,
            "Authorization": f"Basic {auth_key}"
        }
        payload = {
            "scope": self._scope
        }
        try:
            response = await http_client.post(
                url=self._url,
                headers=headers,
                data=payload
            )
            print(response)
        except Exception as _ex:
            raise _ex


import asyncio
from src.http.response.json import JsonResponse
asyncio.run(GigaChatCredentials().get_token(HTTPClient(JsonResponse())))