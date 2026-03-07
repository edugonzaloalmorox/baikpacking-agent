

from collections import Counter
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChunkInfo(BaseModel):
    score: float
    text: str
    chunk_index: Optional[int] = None


class QueryIntent(BaseModel):
    component: str = "full_setup"
    confidence: float = 0.0
    component_terms: List[str] = Field(default_factory=list)
    asks_for_recommendation: bool = True


class RetrievalIntentBundle(BaseModel):
    intent: QueryIntent
    broad_query: str
    component_query: Optional[str] = None
    include_component_query: bool = False


class SimilarRider(BaseModel):
    rider_id: int
    article_id: Optional[int] = None

    name: Optional[str] = None
    event_title: Optional[str] = None
    event_url: Optional[str] = None
    event_key: Optional[str] = None

    frame_type: Optional[str] = None
    frame_material: Optional[str] = None
    wheel_size: Optional[str] = None
    tyre_width: Optional[str] = None
    electronic_shifting: Optional[bool] = None

    best_score: float
    year: Optional[int] = None
    
    bike: Optional[str] = None
    key_items: Optional[str] = None

    chunks: List[ChunkInfo] = Field(default_factory=list)


class SetupCore(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    # Canonical field name is bike_type, but accept "bike" (older prompt/code).
    bike_type: Optional[str] = Field(default=None, alias="bike")

    wheels: Optional[str] = None
    tyres: Optional[str] = None
    drivetrain: Optional[str] = None
    bags: Optional[str] = None
    sleep_system: Optional[str] = None

    # Optional additions (kept optional to avoid “inventing”)
    lighting: Optional[str] = None
    navigation: Optional[str] = None
    water_capacity: Optional[str] = None
    notes: Optional[str] = None

    def is_empty(self) -> bool:
        vals = [
            self.bike_type,
            self.wheels,
            self.tyres,
            self.drivetrain,
            self.bags,
            self.sleep_system,
            self.lighting,
            self.navigation,
            self.water_capacity,
            self.notes,
        ]
        return all(v is None or (isinstance(v, str) and not v.strip()) for v in vals)


class SetupRecommendation(BaseModel):
    """Final grounded agent output."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    event: Optional[str] = None
    summary: str
    reasoning: str

    recommended_setup: SetupCore = Field(default_factory=SetupCore)
    similar_riders: List[SimilarRider] = Field(default_factory=list)

   
    @property
    def bike(self) -> Optional[str]:
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

    @model_validator(mode="after")
    def _ensure_non_empty_setup(self) -> "SetupRecommendation":
        """
        Avoid empty recommendations by generating a minimal, *grounded* fallback
        from SimilarRider structured fields if the model leaves recommended_setup empty.

        This never invents brands/components. It only surfaces what's present in the DB.
        """
        if not self.recommended_setup.is_empty():
            return self

        riders = self.similar_riders or []

        def most_common_str(values):
            vals = [v.strip() for v in values if isinstance(v, str) and v.strip()]
            return Counter(vals).most_common(1)[0][0] if vals else None

        frame_type = most_common_str([r.frame_type for r in riders])
        frame_mat = most_common_str([r.frame_material for r in riders])
        wheel_size = most_common_str([r.wheel_size for r in riders])
        tyre_width = most_common_str([r.tyre_width for r in riders])

        bike_bits = [x for x in [frame_type, frame_mat] if x]
        self.recommended_setup.bike_type = " ".join(bike_bits) if bike_bits else None
        self.recommended_setup.wheels = wheel_size or None

        if tyre_width and wheel_size:
            self.recommended_setup.tyres = f"{tyre_width} on {wheel_size}"
        else:
            self.recommended_setup.tyres = tyre_width or None

        # Leave drivetrain/bags/sleep_system empty if not grounded by chunks.
        return self