import re
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from baikpacking.agents.postprocess import _infer_year_from_title


class CompactChunk(BaseModel):
    chunk_index: Optional[int] = None
    text: str


class CompactRider(BaseModel):
    name: Optional[str] = None
    event_title: Optional[str] = None
    year: Optional[int] = None
    best_score: Optional[float] = None
    bike_type: Optional[str] = None
    wheels: Optional[str] = None
    tyres: Optional[str] = None
    drivetrain: Optional[str] = None
    bags: Optional[str] = None
    sleep_system: Optional[str] = None
    key_items: List[str] = Field(default_factory=list)
    chunks: List[CompactChunk] = Field(default_factory=list)


class WriterInput(BaseModel):
    user_query: str
    event_name: str
    event_context: str
    descriptor_query: str
    query_component: str = "full_setup"
    component_hit_count: int = 0
    similar_riders: List[CompactRider]


def _compact_riders(riders: List[Any]) -> List[CompactRider]:
    out: List[CompactRider] = []

    for r in riders or []:
        compact_chunks: List[CompactChunk] = []
        for idx, c in enumerate(getattr(r, "chunks", []) or []):
            text = getattr(c, "text", None) or getattr(c, "content", None) or ""
            if text:
                compact_chunks.append(
                    CompactChunk(
                        chunk_index=getattr(c, "chunk_index", idx),
                        text=text[:300],
                    )
                )

        key_items = []
        raw_key_items = getattr(r, "key_items", None) or []
        for item in raw_key_items[:5]:
            if isinstance(item, str) and item.strip():
                key_items.append(item.strip())

        out.append(
            CompactRider(
                name=getattr(r, "name", None),
                event_title=getattr(r, "event_title", None),
                year=getattr(r, "year", None) or _infer_year_from_title(getattr(r, "event_title", None)),
                best_score=getattr(r, "best_score", None),
                bike_type=getattr(r, "bike_type", None),
                wheels=getattr(r, "wheels", None),
                tyres=getattr(r, "tyres", None),
                drivetrain=getattr(r, "drivetrain", None),
                bags=getattr(r, "bags", None),
                sleep_system=getattr(r, "sleep_system", None),
                key_items=key_items,
                chunks=compact_chunks[:2],
            )
        )

    return out


def _event_context_to_text(event_context_obj: Any) -> str:
    if not event_context_obj or not getattr(event_context_obj, "context", None):
        return ""

    ctx = event_context_obj.context
    parts = [
        ctx.summary or "",
        ctx.surface or "",
        ctx.route_character or "",
        ctx.climate_notes or "",
        ctx.resupply_notes or "",
        " ".join(ctx.constraints or []),
    ]
    return "\n".join(p for p in parts if p)
