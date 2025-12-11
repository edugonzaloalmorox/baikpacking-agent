import re
from typing import List, Optional

from pydantic_ai import RunContext, Tool

from baikpacking.embedding.qdrant_utils import search_riders_grouped
from baikpacking.agents.models import SimilarRider

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event keywords & helpers duplicated here or imported from a shared place
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


@Tool
def search_similar_riders(
    ctx: RunContext,
    query: str,
    top_k_riders: int = 5,
) -> List[SimilarRider]:
    """
    Search the DotWatcher riders database for the most similar riders to the
    given query.
    ...
    """
    logger.info("search_similar_riders called", extra={"query": query, "top_k_riders": top_k_riders})

    raw = search_riders_grouped(
        query=query,
        top_k_riders=top_k_riders * 2,  # oversample a bit, then re-rank
        oversample_factor=5,
        max_chunks_per_rider=3,
    )

    logger.info("search_riders_grouped returned %d raw riders", len(raw))

    riders = [SimilarRider(**r) for r in raw]

    event_hint = _extract_event_hint(query)

    def sort_key(r: SimilarRider):
        title_lower = (r.event_title or "").lower()
        same_event = 1 if (event_hint and event_hint in title_lower) else 0
        year = r.year or _infer_year_from_title(r.event_title) or 0
        return (same_event, year, r.best_score)

    sorted_riders = sorted(riders, key=sort_key, reverse=True)
    final_riders = sorted_riders[:top_k_riders]

    # Log a short summary of outputs (avoid dumping everything)
    logger.info(
        "search_similar_riders returning %d riders: %s",
        len(final_riders),
        [r.name for r in final_riders if r.name][:3],  # first few names
    )

    return final_riders