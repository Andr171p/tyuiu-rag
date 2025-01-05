from abc import ABC, abstractmethod


class AbstractCredentials(ABC):
    @abstractmethod
    def get_auth_key(self) -> str: pass
