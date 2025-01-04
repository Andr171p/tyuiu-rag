from typing import Dict, Any, Union

from httpx import Response

from src.http.abc.response import AbstractResponse
from src.http.utils.checker import is_ok


class JsonResponse(AbstractResponse):
    def data(
            self,
            response: Response
    ) -> Union[Dict[str, Any]] | None:
        if not is_ok(response):
            return
        return response.json()
