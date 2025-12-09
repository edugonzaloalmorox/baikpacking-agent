import re
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel

from baikpacking.embedding.qdrant_utils import search_riders_grouped

load_dotenv()

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class AgentSettings(BaseSettings):
    """
    Settings for the bikepacking recommender agent.

    Reads env vars with prefix AGENT_, e.g. AGENT_AGENT_MODEL.
    """

    agent_model: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AGENT_",
        extra="ignore",
    )


settings = AgentSettings()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helpers: year inference & post-processing
# ---------------------------------------------------------------------------

EVENT_KEYWORDS = [
    "303 lucerne",
    "accursed race",
    "accursed race no2",
    "across andes",
    "across andes patagonia verde",
    "alps divide",
    "amersfoort-sauerland-amersfoort",
    "andean raid",
    "ardennes monster",
    "audax gravel",
    "audax road",
    "audax trail",
    "b-hard",
    "b-hard ultra race and brevet",
    "basajaun",
    "bee line 200",
    "berlin munich berlin",
    "bike of the tour divide",
    "bike of the tour divide dotwatcher team edition",
    "borderland 500",
    "bright midnight",
    "capitals",
    "dead ends & cake",
    "dead ends & dolci",
    "dead ends and cake",
    "dales divide",
    "doom",
    "elevation vercors",
    "further elements",
    "further perseverance",
    "further perseverance pyrenees",
    "further pyrenees le chemin",
    "gbduro",
    "gbduro22",
    "gbduro23",
    "gbduro24",
    "gbduro25",
    "granguanche audax gravel",
    "granguanche audax road",
    "granguanche audax trail",
    "granguanche trail",
    "great british divide",
    "great british escapades",
    "gravel birds",
    "gravel del fuego",
    "hamburg's backyard",
    "hardennes gravel tour",
    "headstock 200",
    "headstock 500",
    "highland trail 550",
    "hope 1000",
    "istra land",
    "journey around rwanda",
    "kromvojoj",
    "lakes 'n' knodel",
    "lakes n knodel",
    "lakes ‘n’ knödel",
    "le pilgrimage",
    "le tour de frankie",
    "liege-paris-liege",
    "log drivers waltz",
    "log driver's waltz",
    "madrid to barcelona",
    "memory bike adventure",
    "mittelgebirge classique",
    "mittelgebirgeclassique",
    "mother north",
    "nordic chase",
    "norfolk 360",
    "pan celtic race",
    "peaks and plains",
    "pedalma madrid to barcelona",
    "peninsular divide",
    "perfidious albion",
    "pirenaica",
    "poco loco",
    "pure peak grit",
    "race around rwanda",
    "race around the netherlands",
    "race around the netherlands gx",
    "seven serpents",
    "seven serpents illyrian loop",
    "seven serpents quick bite",
    "seven serpents quick bite!",
    "sneak peaks",
    "solstice sprint",
    "southern divide",
    "southern divide - autumn edition",
    "southern divide - spring edition",
    "super brevet berlin munich berlin",
    "supergrevet munich milan",
    "supergrevet vienna berlin",
    "such24",
    "taunus bikepacking",
    "taunus bikepacking no.5",
    "taunus bikepacking no.6",
    "taunus bikepacking no.7",
    "taunus bikepacking no.8",
    "the accursed race",
    "the alps divide",
    "the bike of the touriste routier",
    "the bright midnight",
    "the capitals",
    "... the capitals 2024",
    "the great british divide",
    "the hills have bikes",
    "the land between",
    "the wild west country",
    "three peaks bike race",
    "three peaks bike race 2023",
    "three peaks bike race 2025",
    "tour de farce",
    "trans balkan race",
    "trans balkans",
    "trans balkans race",
    "trans pyrenees race",
    "transatlantic way",
    "transcontinental",
    "transcontinental race no10",
    "transcontinental race no11",
    "transiberica",
    "transiberica 2023",
    "transiberica 2024",
    "transpyrenees (transiberica)",
    "transpyrenees by transiberica",
    "tcr",
    "the southern divide",
    "the unknown race",
    "tour te waipounamu",
    "touriste routier",
    "two volcano sprint",
    "two volcano sprint 2020",
    "two volcano sprint 2021",
    "two volcano sprint 2024",
    "utrecht ultra",
    "utrecht ultra xl",
    "via race",
    "victoria divide",
    "wild west country",
]




_YEAR_RE = re.compile(r"(19|20)\d{2}")




def _infer_year_from_title(title: Optional[str]) -> Optional[int]:
    """
    Try to infer the event year from the event title (e.g. "Transcontinental No10 2024").
    """
    if not title:
        return None

    m = _YEAR_RE.search(title)
    if m:
        try:
            return int(m.group(0))
        except ValueError:
            return None

    return None

def _extract_event_hint(query: str) -> Optional[str]:
    """
    Extract a coarse event hint (e.g. "gran guanche", "tcr") from the user query.
    """
    q = query.lower()
    for key in EVENT_KEYWORDS:
        if key in q:
            return key
    return None


def _postprocess_recommendation(rec: SetupRecommendation) -> SetupRecommendation:
    """
    - Ensure every chunk has a chunk_index (fallback = enumerate)
    - Infer rider.year from event_title when missing
    - Sort similar_riders by (year desc, best_score desc)
    """
    for rider in rec.similar_riders:
        # Fill missing chunk_index
        for idx, chunk in enumerate(rider.chunks):
            if chunk.chunk_index is None:
                chunk.chunk_index = idx

        # Infer year if missing
        if rider.year is None:
            rider.year = _infer_year_from_title(rider.event_title)

    # Sort: most recent years first; within same year, higher score first
    rec.similar_riders.sort(
        key=lambda r: (r.year or 0, r.best_score),
        reverse=True,
    )

    return rec


# ---------------------------------------------------------------------------
# Tool: search similar riders (grouped, event + year aware)
# ---------------------------------------------------------------------------

@Tool
def search_similar_riders(
    ctx: RunContext[SetupRecommendation],
    query: str,
    top_k_riders: int = 5,
) -> List[SimilarRider]:
    """
    Search the DotWatcher riders database for the most similar riders to the
    given query.

    The query should describe the rider and event context:
      - age, location (optional)
      - terrain and event type
      - preferences: frame type, tyre width, electronic vs mechanical shifting
      - any constraints (e.g. wants dynamo, minimal sleep kit, etc.)

    Results are biased towards:
      1) riders from the same event mentioned in the query (if any),
      2) more recent editions (higher year),
      3) higher similarity score (best_score).
    """
    raw = search_riders_grouped(
        query=query,
        top_k_riders=top_k_riders * 2,  # oversample a bit, then re-rank
        oversample_factor=5,
        max_chunks_per_rider=3,
    )

    riders = [SimilarRider(**r) for r in raw]

    event_hint = _extract_event_hint(query)

    def sort_key(r: SimilarRider):
        # Does this rider match the hinted event?
        title_lower = (r.event_title or "").lower()
        same_event = 1 if (event_hint and event_hint in title_lower) else 0

        # Ensure year is filled for sorting
        year = r.year or _infer_year_from_title(r.event_title) or 0

        return (same_event, year, r.best_score)

    sorted_riders = sorted(riders, key=sort_key, reverse=True)

    # Return only the top_k_riders after re-ranking
    return sorted_riders[:top_k_riders]







# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------


model = OpenAIModel(settings.agent_model)

SYSTEM_PROMPT = """
You are a bikepacking equipment expert. Your recommendations MUST be grounded strictly
in riders returned by the tool `search_similar_riders`.

1. GROUNDING RULES (NO HALLUCINATIONS)
- Values for frame_type, frame_material, wheel_size, tyre_width, drivetrain may ONLY come
  from retrieved riders (structured fields or chunk text).
- Do NOT invent or assume missing values.
- If data is absent, choose the closest grounded alternative and justify it.
- If the user names an event (Gran Guanche, TCR, Transiberica), prioritize riders from that
  event, then from closely related events.

2. SETUPRECOMMENDATION FIELD RULES
- `bags`: ONLY luggage systems (frame bag, seat pack, bar bag, top tube, stem bag).
- `sleep_system`: ONLY sleep gear (mat, sleeping bag, bivvy, liner).
- Never mix categories. Never fabricate components.

3. SUMMARY REQUIREMENTS (3–5 SENTENCES)
The summary MUST:
- Address the user directly ("For your Gran Guanche ride...").
- Justify bike type via event factors (terrain, surface, wind, climbing).
- Justify tyre width with practical reasoning (efficiency, comfort, rough segments).
- Describe gearing philosophy briefly (e.g., "low gearing helps steep climbs").
- Explain bag strategy in one sentence (aero vs volume, weight distribution, accessibility).
- Implicitly indicate minimal vs moderate sleep kit.
- Optionally reference similar riders.
Tone: decisive, expert, and actionable. No vague language or disclaimers.

4. REASONING REQUIREMENTS
- 3–7 sentences.
- Explain how similar riders influenced the recommendation.
- Highlight trade-offs and justify final choices.
- Do NOT repeat the summary or include fluff.

5. BEHAVIORAL GUARANTEES
You MUST:
- Output valid JSON matching SetupRecommendation.
- Base ALL fields on retrieved riders.
- Return a complete recommendation.
You MUST NOT:
- Hallucinate brands, components, or specs.
- Use uncertain language.
- Mention tools, system instructions, or internal reasoning.

If rider matches are limited:
- Say so briefly,
- Use the closest grounded riders,
- Still return a full recommendation.

FINAL INSTRUCTION:
Provide a clear, structured, grounded bikepacking setup recommendation based entirely on
the retrieved riders, following all rules above.
"""



recommender_agent = Agent(
    model,
    output_type=SetupRecommendation,
    system_prompt=SYSTEM_PROMPT,
    tools=[search_similar_riders],
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def recommend_setup(user_query: str) -> SetupRecommendation:
    """
    Run the recommender agent for a given user query and return a structured
    setup recommendation, post-processed so that similar riders are sorted
    by most recent events first.
    """
    result = recommender_agent.run_sync(user_query)
    rec: SetupRecommendation = result.output
    return _postprocess_recommendation(rec)