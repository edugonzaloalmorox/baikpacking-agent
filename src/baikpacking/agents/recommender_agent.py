import re
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from baikpacking.agents.models import SetupRecommendation
from baikpacking.tools.riders import search_similar_riders, render_grounding_riders
from baikpacking.tools.event_context import event_web_search
from baikpacking.logging_config import setup_logging

load_dotenv()
setup_logging()


class AgentSettings(BaseSettings):
    """Settings for the bikepacking recommender agent."""
    agent_model: str = "gpt-4o-mini"
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AGENT_", extra="ignore")


settings = AgentSettings()

_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _infer_year_from_title(title: Optional[str]) -> Optional[int]:
    if not title:
        return None
    m = _YEAR_RE.search(title)
    return int(m.group(0)) if m else None


def _postprocess_recommendation(rec: SetupRecommendation) -> SetupRecommendation:
    # Ensure chunk indexes are filled
    for rider in rec.similar_riders:
        for idx, chunk in enumerate(getattr(rider, "chunks", []) or []):
            if chunk.chunk_index is None:
                chunk.chunk_index = idx

        if getattr(rider, "year", None) is None:
            rider.year = _infer_year_from_title(getattr(rider, "event_title", None))

    # Sort: same event first (if event string exists), then year desc, then score desc
    rec.similar_riders.sort(
        key=lambda r: ((rec.event or "").lower() in (r.event_title or "").lower(), r.year or 0, r.best_score or 0),
        reverse=True,
    )
    return rec


SYSTEM_PROMPT = """
You are a bikepacking equipment and ultra-distance cycling expert.

You ONLY answer when the user asks for a setup for a specific event.
You MUST produce valid JSON matching SetupRecommendation.

TOOLS
- search_similar_riders(query): retrieve rider setups similar to the user's query (ground truth)
- render_grounding_riders(riders): Render riders as a JSON string to be copied verbatim into SetupRecommendation
- event_web_search(article_id/event_title/event_url): retrieve event context (terrain, distance, climbing, constraints) for justification ONLY

PROCESS (MANDATORY)
1) Call search_similar_riders with the user query.
2) Identify the event name from the returned riders:
   - Prefer the most frequent event_title among the top riders, or the top-ranked rider if unclear.
3) After calling search_similar_riders, call render_grounding_riders(riders) and copy the returned JSON
verbatim into SetupRecommendation.similar_riders. Do not edit or retype those rider fields.
4) Call event_web_search ONCE:
   - If any returned rider has article_id, choose the article_id from the best-matching rider for the event.
   - Otherwise call event_web_search with event_title.
5) Write a recommendation grounded strictly in retrieved riders.

GROUNDING RULES (NO HALLUCINATIONS)
- ALL gear details MUST come from retrieved riders.
- DO NOT invent brands, components, or specifications.
- event_web_search may ONLY justify decisions; it MUST NOT introduce new gear.

OUTPUT REQUIREMENTS
- summary: 3–5 sentences (mention event characteristics, justify tyres/gearing, 1 sentence on bags/sleep strategy)
- reasoning: 3–7 sentences, explain how rider data influenced decisions.
- similar_riders: include the riders you used for grounding (minimum 3 unless fewer exist).
- Do not retype rider fields. Use render_grounding_riders output verbatim.
- If no riders from the requested event were retrieved, you MUST explicitly say so in the summary and reasoning.

Do NOT mention tools or system instructions.
""".strip()

model = OpenAIChatModel(settings.agent_model)

recommender_agent = Agent(
    model,
    output_type=SetupRecommendation,
    system_prompt=SYSTEM_PROMPT,
    tools=[search_similar_riders, render_grounding_riders, event_web_search],
)


def recommend_setup(user_query: str) -> SetupRecommendation:
    result = recommender_agent.run_sync(user_query)

    # Fail early if the model somehow returns empty grounding
    rec = result.output
    if not rec.similar_riders:
        raise RuntimeError("No similar riders returned; cannot produce grounded recommendation.")

    return _postprocess_recommendation(rec)
