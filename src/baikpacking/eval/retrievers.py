from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedHit:
    doc_id: int
    score: float
    point_id: int
    payload: Optional[Dict[str, Any]] = None


class BaseRetriever(ABC):
    name: str

    @abstractmethod
    def search(self, query: str, k: int) -> List[RetrievedHit]:
        raise NotImplementedError