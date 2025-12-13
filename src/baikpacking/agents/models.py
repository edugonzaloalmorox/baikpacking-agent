from collections import Counter
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class ChunkInfo(BaseModel):
    score: float
    text: str
    chunk_index: Optional[int] = None


class SimilarRider(BaseModel):
    rider_id: int
    article_id: Optional[int] = None  # if you want event_web_search by DB id

    name: str
    event_title: Optional[str] = None
    event_url: Optional[str] = None
    event_key: Optional[str] = None  # helpful for strict event grounding

    frame_type: Optional[str] = None
    frame_material: Optional[str] = None
    wheel_size: Optional[str] = None
    tyre_width: Optional[str] = None
    electronic_shifting: Optional[bool] = None

    best_score: float
    year: Optional[int] = None

    chunks: List[ChunkInfo] = Field(default_factory=list)


class SetupCore(BaseModel):
    bike_type: Optional[str] = None
    wheels: Optional[str] = None
    tyres: Optional[str] = None
    drivetrain: Optional[str] = None
    bags: Optional[str] = None
    sleep_system: Optional[str] = None


class SetupRecommendation(BaseModel):
    event: Optional[str] = None
    summary: str
    reasoning: Optional[str] = None

    recommended_setup: SetupCore = Field(default_factory=SetupCore)
    similar_riders: List[SimilarRider] = Field(default_factory=list)

    # -----------------
    # Convenience accessors (for CLI / UI)
    # -----------------
    @property
    def bike_type(self) -> Optional[str]:
        return self.recommended_setup.bike_type

    @property
    def wheels(self) -> Optional[str]:
        return self.recommended_setup.wheels

    @property
    def tyres(self) -> Optional[str]:
        return self.recommended_setup.tyres

    @property
    def drivetrain(self) -> Optional[str]:
        return self.recommended_setup.drivetrain

    @property
    def bags(self) -> Optional[str]:
        return self.recommended_setup.bags

    @property
    def sleep_system(self) -> Optional[str]:
        return self.recommended_setup.sleep_system

    # -----------------
    # Validators
    # -----------------
    @model_validator(mode="after")
    def _validate_grounding_and_event(self):
        if not self.similar_riders:
            raise ValueError("similar_riders must be non-empty for grounding.")

        if not self.event:
            titles = [r.event_title for r in self.similar_riders if r.event_title]
            if titles:
                self.event = Counter(titles).most_common(1)[0][0]
            else:
                raise ValueError("event is required (could not infer from similar_riders).")
        return self

    @model_validator(mode="after")
    def _no_empty_setup(self):
        s = self.recommended_setup
        if not any([s.bike_type, s.wheels, s.tyres, s.drivetrain, s.bags, s.sleep_system]):
            raise ValueError("recommended_setup is empty; fallback logic failed")
        return self
