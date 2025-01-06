from abc import ABC, abstractmethod
from typing import Union, Dict, Any

from httpx import Response


class AbstractResponse(ABC):
    @abstractmethod
    def data(
            self,
            response: Response
    ) -> Union[Dict[str, Any]] | None: pass
