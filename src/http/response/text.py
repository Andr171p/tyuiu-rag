from httpx import Response

from src.http.abc.response import AbstractResponse
from src.http.utils.checker import is_ok


class TextResponse(AbstractResponse):
    def data(
            self,
            response: Response
    ) -> str | None:
        if not is_ok(response):
            return
        return response.text