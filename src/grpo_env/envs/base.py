from typing import Protocol


class Env(Protocol):
    def load(self, split: str) -> list[dict]:
        ...
