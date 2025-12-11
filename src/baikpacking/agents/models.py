from typing import List, Optional
from pydantic import BaseModel, Field


class ChunkInfo(BaseModel):
    """
    Represents a single matched text chunk for a rider.
    """

    score: float
    text: str
    chunk_index: Optional[int] = None


class SimilarRider(BaseModel):
    """
    A rider used as inspiration for the recommendation.
    """

    rider_id: int

    name: Optional[str] = None
    event_title: Optional[str] = None
    event_url: Optional[str] = None

    frame_type: Optional[str] = None
    frame_material: Optional[str] = None
    wheel_size: Optional[str] = None
    tyre_width: Optional[str] = None
    electronic_shifting: Optional[bool] = None

    best_score: float
    year: Optional[int] = None

    chunks: List[ChunkInfo] = Field(default_factory=list)


class SetupRecommendation(BaseModel):
    """
    Final output of the recommender agent.

    - event: target event (e.g. "Transcontinental No10")
    - bike_type: headline bike description
    - wheels / tyres / drivetrain: key drivetrain choices
    - bags: luggage system (frame / seat / bar / top tube / stem bags)
    - sleep_system: sleeping gear (mat, bag, bivvy, etc.)
    - summary: short, user-facing summary
    - reasoning: explanation grounded in similar riders
    - similar_riders: list of riders that inspired this setup
    """

    event: Optional[str] = None

    bike_type: Optional[str] = None
    wheels: Optional[str] = None
    tyres: Optional[str] = None
    drivetrain: Optional[str] = None

    bags: Optional[str] = None
    sleep_system: Optional[str] = None

    summary: str
    reasoning: Optional[str] = None

    similar_riders: List[SimilarRider] = Field(default_factory=list)