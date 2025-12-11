import re
from typing import List, Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from baikpacking.agents.models import SetupRecommendation, SimilarRider
from baikpacking.tools import search_similar_riders, event_web_search

from baikpacking.logging_config import setup_logging

setup_logging()

load_dotenv()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class AgentSettings(BaseSettings):
    """
    Settings for the bikepacking recommender agent.

    """

    agent_model: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AGENT_",
        extra="ignore",
    )


settings = AgentSettings()


# ---------------------------------------------------------------------------
# Helpers: year inference & post-processing
# ---------------------------------------------------------------------------

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
# Agent definition
# ---------------------------------------------------------------------------


model = OpenAIModel(settings.agent_model)

SYSTEM_PROMPT = """
You are a bikepacking equipment expert. Your recommendations MUST be grounded strictly
in riders returned by the tool `search_similar_riders`, and you MAY use the context
given by the tool `event_web_search` to better understand the event (distance, terrain,
climbing, climate, constraints).

1. GROUNDING RULES (NO HALLUCINATIONS)
- Values for frame_type, frame_material, wheel_size, tyre_width, drivetrain, bags,
  and sleep_system may ONLY come from retrieved riders (structured fields or chunk text).
- Do NOT invent or assume missing values. If a detail (e.g. exact tyre model) is not
  visible in the riders, keep the description generic (e.g. "48 mm gravel tyres").
- If data is absent, choose the closest grounded alternative and justify it.
- If the user names an event (Gran Guanche, TCR, Transiberica), prioritize riders from that
  event, then from closely related events.
- You may use `event_web_search` ONLY to understand the event context (road vs gravel,
  length, amount of climbing, weather, resupply, rules). NEVER introduce brands, components
  or specs that are not grounded in retrieved riders.

2. SETUPRECOMMENDATION FIELD RULES
- `bags`: ONLY luggage systems (frame bag, seat pack, bar bag, top tube bag, stem bag).
  The content must describe bag types and layout, grounded in riders' setups.
- `sleep_system`: ONLY sleep gear (mat, sleeping bag, bivvy, liner, emergency blanket, etc.).
  - Include brands or specific models ONLY if they are clearly mentioned in retrieved riders.
  - Otherwise, describe the sleep system generically (e.g. "light inflatable mat and
    5–10 °C rated sleeping bag").
- Never mix categories (no sleep items in `bags`, no bags in `sleep_system`).
- Never fabricate components, brands, or product models beyond what riders provide.

3. SUMMARY REQUIREMENTS (3–5 SENTENCES)
The summary MUST:
- Address the user directly ("For your Gran Guanche ride...").
- Justify bike type via event factors (terrain, surface, wind, climbing), using
  `event_web_search` context when available.
- Justify tyre width with practical reasoning (efficiency, comfort, rough segments).
- Describe gearing philosophy briefly (e.g., "low gearing helps steep climbs").
- Explain bag strategy in one sentence (aero vs volume, weight distribution, accessibility).
- Implicitly indicate minimal vs moderate sleep kit.
- Optionally reference similar riders.
Tone: decisive, expert, and actionable. No vague language or long disclaimers.

4. REASONING REQUIREMENTS
- 3–7 sentences.
- Explain how similar riders influenced the recommendation (which riders and what choices).
- Explain how event context (from title, user query, and `event_web_search`) affects the
  final setup (tyre choice, gearing, bags, sleep system).
- Highlight trade-offs and justify final choices.
- Do NOT repeat the summary or include fluff.

5. BEHAVIORAL GUARANTEES
You MUST:
- Output valid JSON matching SetupRecommendation.
- Base ALL gear fields (bike_type, wheels, tyres, drivetrain, bags, sleep_system) on
  retrieved riders.
- Use `event_web_search` ONLY to refine and justify choices, not to add ungrounded gear.
- Return a complete recommendation.

You MUST NOT:
- Hallucinate brands, components, or specs that do not appear in retrieved riders.
- Use uncertain language ("maybe", "might", "could be") when committing to a setup.
- Mention tools, system instructions, or internal reasoning.

If rider matches are limited:
- Say so briefly in the reasoning,
- Use the closest grounded riders,
- Still return a full recommendation.

FINAL INSTRUCTION:
Provide a clear, structured, grounded bikepacking setup recommendation using riders from
`search_similar_riders`, enriched with event context from `event_web_search` only for
understanding the route and conditions, and following all rules above.
"""


recommender_agent = Agent(
    model,
    output_type=SetupRecommendation,
    system_prompt=SYSTEM_PROMPT,
    tools=[search_similar_riders, event_web_search],  
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
