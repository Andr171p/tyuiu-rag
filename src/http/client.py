from typing import Optional, Dict, Any

from httpx import AsyncClient

from src.http.abstract.client import AbstractClient
from src.http.abstract.response import AbstractResponse


class HTTPClient(AbstractClient):
    def __init__(self, response_type: AbstractResponse) -> None:
        self._response_type = response_type

    async def get(
            self,
            url: str,
            headers: Optional[Dict[str, Any]] = None,
            ssl: bool = False
    ) -> Any | None:
        async with AsyncClient(
            http2=True,
            verify=ssl
        ) as client:
            response = await client.get(
                url=url,
                headers=headers
            )
            return self._response_type.data(response)

    async def post(
            self,
            url: str,
            data: Dict[str, Any],
            headers: Optional[Dict[str, Any]] = None,
            ssl: bool = False
    ) -> Any | None:
        async with AsyncClient(
            http2=True,
            verify=ssl
        ) as client:
            response = await client.post(
                url=url,
                data=data,
                headers=headers
            )
            return self._response_type.data(response)
