from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EventResolutionResult(BaseModel):
    raw_query_event: Optional[str] = None
    canonical_name: Optional[str] = None
    display_name: str = "Unknown event"
    requested_count: Optional[int] = None
    match_type: str = "unknown"
    confidence: float = 0.0
    is_trusted_exact: bool = False


class EventContextSummary(BaseModel):
    requested_event_name: str
    web_context_text: str = ""
    similar_events: List[str] = Field(default_factory=list)
    event_family: Optional[str] = None
    family_confidence: float = 0.0
    archetype: Optional[str] = None
    surface_family: Optional[str] = None
    features: Dict[str, Any] = Field(default_factory=dict)


class RetrievalPlan(BaseModel):
    query_component: str = "full_setup"
    use_exact_event: bool = False
    event_name_for_retrieval: Optional[str] = None
    descriptor_query: str
    descriptor_query_with_intent: Optional[str] = None
    primary_query: str
    fallback_query: Optional[str] = None
    fallback_reasoning: Optional[str] = None
    intent_bundle: Any = None


class RetrievalExecutionResult(BaseModel):
    riders: List[Any] = Field(default_factory=list)
    used_query: str
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    retrieval_source: str = "unknown_global"
    exact_event_hit_count: int = 0
    matched_event_name: Optional[str] = None
    component_hit_count: int = 0


class EvidenceSummary(BaseModel):
    rider_count: int = 0
    component_hit_count: int = 0
    field_support: Dict[str, str] = Field(default_factory=dict)
    evidence_strength: str = "none"
    consistency: str = "unknown"


class RecommendationPolicy(BaseModel):
    mode: str = "strict_grounded"
    allow_specific_brands: bool = True
    allow_specific_specs: bool = True
    allow_event_specific_claims: bool = True
    notes: List[str] = Field(default_factory=list)


class RecommendationRunDiagnostics(BaseModel):
    event_resolution: Optional[EventResolutionResult] = None
    event_context: Optional[EventContextSummary] = None
    retrieval_plan: Optional[RetrievalPlan] = None
    evidence_summary: Optional[EvidenceSummary] = None
    policy: Optional[RecommendationPolicy] = None
    fallback_used: bool = False
