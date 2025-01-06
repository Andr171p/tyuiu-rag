from abc import ABC, abstractmethod


class AbstractAuth(ABC):
    @abstractmethod
    def get_auth_key(self) -> str:
        raise NotImplementedError("get_auth_key method is not implemented")
