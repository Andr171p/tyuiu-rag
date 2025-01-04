from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class AbstractClient(ABC):
    @abstractmethod
    async def get(
            self,
            url: str,
            headers: Optional[Dict[str, Any]] = None,
            ssl: bool = False
    ) -> Any | None: pass

    @abstractmethod
    async def post(
            self,
            url: str,
            data: Dict[str, Any],
            headers: Optional[Dict[str, Any]] = None,
            ssl: bool = False
    ) -> Any | None: pass
