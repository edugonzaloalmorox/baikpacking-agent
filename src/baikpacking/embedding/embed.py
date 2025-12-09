import os
import json
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional

import requests

from .config import Settings


settings = Settings()

# ---------------------------------------------------------------------------
# Event aliases → used to derive a clean event_key from article titles
# ---------------------------------------------------------------------------

EVENT_ALIASES: Dict[str, List[str]] = {

    # ----------------------- GranGuanche -----------------------
    "granguanche-audax-trail": [
        "granguanche audax trail", "granguanche trail", "audax trail"
    ],
    "granguanche-audax-gravel": [
        "granguanche audax gravel", "audax gravel"
    ],
    "granguanche-audax-road": [
        "granguanche audax road", "audax road"
    ],

    # ----------------------- Further ---------------------------
    "further-elements": ["further elements"],
    "further-perseverance": ["further perseverance"],
    "further-perseverance-pyrenees": ["further perseverance pyrenees"],
    "further-pyrenees-le-chemin": ["further pyrenees le chemin"],

    # ----------------------- Peninsular Divide -----------------
    "peninsular-divide": ["peninsular divide"],

    # ----------------------- Tour de Farce ----------------------
    "tour-de-farce": ["tour de farce"],

    # ----------------------- Trans Pyrenees ---------------------
    "trans-pyrenees-race": ["trans pyrenees race", "transpyrenees"],
    "transpyrenees-transiberica": ["transpyrenees by transiberica", "transpyrenees (transiberica)"],

    # ----------------------- Pirenaica --------------------------
    "pirenaica": ["pirenaica"],

    # ----------------------- Istra Land -------------------------
    "istra-land": ["istra land"],

    # ----------------------- Bohemian Border Bash ---------------
    "bohemian-border-bash-race": ["bohemian border bash race"],

    # ----------------------- Lakes 'n' Knödel -------------------
    "lakes-n-knodel": ["lakes 'n' knodel", "lakes n knodel", "lakes ‘n’ knödel"],

    # ----------------------- Sneak Peaks ------------------------
    "sneak-peaks": ["sneak peaks"],

    # ----------------------- Supergrevet ------------------------
    "supergrevet-vienna-berlin": ["supergrevet vienna berlin"],
    "supergrevet-berlin-munich-berlin": ["super brevet berlin munich berlin"],
    "supergrevet-munich-milan": ["supergrevet munich milan"],

    # ----------------------- Log Drivers Waltz ------------------
    "log-drivers-waltz": ["log drivers waltz", "log driver's waltz"],

    # ----------------------- The Land Between -------------------
    "the-land-between": ["the land between"],

    # ----------------------- Ardennes Monster -------------------
    "ardennes-monster": ["ardennes monster"],

    # ----------------------- Bentang Jawa -----------------------
    "bentang-jawa": ["bentang jawa"],

    # ----------------------- GBDURO -----------------------------
    "gbduro": ["gbduro", "gbduro24", "gbduro25", "gbduro23", "gbduro22"],

    # ----------------------- Berlin–Munich–Berlin ---------------
    "berlin-munich-berlin": ["berlin munich berlin"],

    # ----------------------- Basajaun ---------------------------
    "basajaun": ["basajaun"],

    # ----------------------- VIA Race ---------------------------
    "via-race": ["via race"],

    # ----------------------- Transcontinental -------------------
    "transcontinental": ["transcontinental", "tcr", "transcontinental race no10", "transcontinental race no11"],

    # ----------------------- Hills Have Bikes -------------------
    "hills-have-bikes": ["the hills have bikes"],

    # ----------------------- Utrecht Ultra ----------------------
    "utrecht-ultra": ["utrecht ultra", "utrecht ultra xl"],

    # ----------------------- Capitals by Pedalma ----------------
    "capitals-by-pedalma": [
        "the capitals by pedalma",
        "the capitals",
        "capitals",
        "... the capitals 2024"
    ],

    # ----------------------- Three Peaks Bike Race --------------
    "three-peaks-bike-race": [
        "three peaks bike race",
        "three peaks bike race 2023",
        "three peaks bike race 2025"
    ],

    # ----------------------- Bright Midnight --------------------
    "bright-midnight": ["bright midnight", "the bright midnight"],

    # ----------------------- Peak Grit --------------------------
    "pure-peak-grit": ["pure peak grit"],

    # ----------------------- Andean Raid ------------------------
    "andean-raid": ["andean raid"],

    # ----------------------- Dead Ends & Cake -------------------
    "dead-ends-and-cake": ["dead ends and cake", "dead ends & dolci", "dead ends & cake"],

    # ----------------------- Solstice Sprint --------------------
    "solstice-sprint": ["solstice sprint"],

    # ----------------------- Bike of the Tour Divide ------------
    "bike-of-tour-divide": [
        "bike of the tour divide",
        "bike of the tour divide dotwatcher team edition"
    ],

    # ----------------------- Taunus Bikepacking -----------------
    "taunus-bikepacking": [
        "taunus bikepacking",
        "taunus bikepacking no.7",
        "taunus bikepacking no.8",
        "taunus bikepacking no.6",
        "taunus bikepacking no.5"
    ],

    # ----------------------- Touriste Routier -------------------
    "touriste-routier": ["the bike of the touriste routier"],

    # ----------------------- Race Around The Netherlands --------
    "race-around-netherlands": [
        "race around the netherlands",
        "race around the netherlands gx",
    ],

    # ----------------------- Nordic Chase -----------------------
    "nordic-chase": ["nordic chase"],

    # ----------------------- Hamburg’s Backyard -----------------
    "hamburgs-backyard": ["hamburg's backyard"],

    # ----------------------- Mittelgebirge Classique ------------
    "mittelgebirge-classique": [
        "mittelgebirge classique",
        "mittelgebirgeclassique"
    ],

    # ----------------------- Amersfoort-Sauerland ---------------
    "amersfoort-sauerland-amersfoort": [
        "amersfoort-sauerland-amersfoort"
    ],

    # ----------------------- Trans Balkan ------------------------
    "trans-balkan-race": [
        "trans balkan race",
        "trans balkans race",
        "trans balkans"
    ],

    # ----------------------- Great British Escapades ------------
    "great-british-escapades": ["great british escapades"],

    # ----------------------- Hardennes Gravel Tour --------------
    "hardennes-gravel-tour": ["hardennes gravel tour"],

    # ----------------------- Pedalma M2B -------------------------
    "madrid-to-barcelona": [
        "pedalma madrid to barcelona",
        "madrid to barcelona"
    ],

    # ----------------------- Headstock --------------------------
    "headstock-500": ["headstock 500"],
    "headstock-200": ["headstock 200"],

    # ----------------------- Highland Trail ---------------------
    "highland-trail-550": ["highland trail 550", "the highland trail 550"],

    # ----------------------- Peaks and Plains -------------------
    "peaks-and-plains": ["peaks and plains"],

    # ----------------------- Seven Serpents ----------------------
    "seven-serpents": [
        "seven serpents",
        "seven serpents quick bite",
        "seven serpents quick bite!",
        "seven serpents illyrian loop"
    ],

    # ----------------------- Bee Line 200 ------------------------
    "bee-line-200": ["bee line 200"],

    # ----------------------- 303 Lucerne -------------------------
    "303-lucerne": ["303 lucerne"],

    # ----------------------- The Accursed Race ------------------
    "accursed-race": [
        "the accursed race",
        "the accursed race no2"
    ],

    # ----------------------- Wild West Country -------------------
    "wild-west-country": ["the wild west country", "wild west country"],

    # ----------------------- Southern Divide ---------------------
    "southern-divide": [
        "the southern divide", 
        "the southern divide - spring edition",
        "the southern divide - autumn edition"
    ],

    # ----------------------- Gravel Birds -----------------------
    "gravel-birds": ["gravel birds"],

    # ----------------------- Dales Divide ------------------------
    "dales-divide": ["dales divide"],

    # ----------------------- Le Tour de Frankie -----------------
    "le-tour-de-frankie": ["le tour de frankie"],

    # ----------------------- Unknown Race ------------------------
    "unknown-race": ["the unknown race"],

    # ----------------------- Norfolk 360 -------------------------
    "norfolk-360": ["norfolk 360"],

    # ----------------------- Doom -------------------------------
    "doom": ["doom"],

    # ----------------------- Race Around Rwanda -----------------
    "race-around-rwanda": ["race around rwanda"],

    # ----------------------- Across Andes ------------------------
    "across-andes": ["across andes", "across andes patagonia verde"],

    # ----------------------- Two Volcano Sprint ------------------
    "two-volcano-sprint": [
        "two volcano sprint",
        "two volcano sprint 2021",
        "two volcano sprint 2024",
        "two volcano sprint 2020"
    ],

    # ----------------------- Borderland 500 ----------------------
    "borderland-500": ["borderland 500"],

    # ----------------------- Le Pilgrimage ------------------------
    "le-pilgrimage": ["le pilgrimage"],

    # ----------------------- SUCH24 ------------------------------
    "such24": ["such24"],

    # ----------------------- Alps Divide -------------------------
    "alps-divide": ["alps divide", "the alps divide"],

    # ----------------------- TransIberica -------------------------
    "transiberica": ["transiberica", "transibérica", "transiberica 2023", "transiberica 2024"],

    # ----------------------- Liege–Paris–Liege -------------------
    "liege-paris-liege": ["liège-paris-liège", "liege-paris-liege"],

    # ----------------------- Perfidious Albion -------------------
    "perfidious-albion": ["the perfidious albion"],

    # ----------------------- Great British Divide ----------------
    "great-british-divide": [
        "great british divide",
        "the great british divide"
    ],

    # ----------------------- Colorado Trail Race -----------------
    "colorado-trail-race": ["colorado trail race"],

    # ----------------------- Mother North ------------------------
    "mother-north": ["mother north"],

    # ----------------------- TransAtlantic Way -------------------
    "transatlantic-way": ["transatlantic way", "the transatlantic way"],

    # ----------------------- Pan Celtic --------------------------
    "pan-celtic-race": ["pan celtic race"],

    # ----------------------- Elevation Vercors -------------------
    "elevation-vercors": ["elevation vercors"],

    # ----------------------- Memory Bike Adventure ---------------
    "memory-bike-adventure": ["memory bike adventure"],

    # ----------------------- Hope 1000 ---------------------------
    "hope-1000": ["hope 1000"],

    # ----------------------- Blaenau 600 -------------------------
    "blaenau-600": ["blaenau 600"],

    # ----------------------- B-HARD Ultra ------------------------
    "bhard-ultra": ["b-hard ultra race and brevet", "b-hard"],

    # ----------------------- Kromvojoj ---------------------------
    "kromvojoj": ["kromvojoj"],

    # ----------------------- Unmapping Sweden --------------------
    "unmapping-sweden": ["unmapping sweden", "unmapping: sweden"],

    # ----------------------- Gravel del Fuego --------------------
    "gravel-del-fuego": ["gravel del fuego"],

    # ----------------------- Poco Loco ---------------------------
    "poco-loco": ["poco loco"],

    # ----------------------- Journey Around Rwanda ---------------
    "journey-around-rwanda": ["journey around rwanda"],

    # ----------------------- Victoria Divide ---------------------
    "victoria-divide": ["victoria divide"],
}




def _normalize_text_for_match(s: str) -> str:
    """
    Lowercase, remove accents and keep only alphanumerics + spaces.
    Helps match 'TransIbérica' with 'transiberica'.
    """
    if not s:
        return ""

    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

    cleaned: List[str] = []
    for ch in s:
        if ch.isalnum() or ch.isspace():
            cleaned.append(ch)
    return "".join(cleaned)


def infer_event_key_from_title(title: str) -> Optional[str]:
    """
    Infer a canonical event_key from the article title using EVENT_ALIASES.
    """
    norm = _normalize_text_for_match(title)
    if not norm:
        return None

    for key, aliases in EVENT_ALIASES.items():
        for alias in aliases:
            if alias in norm:
                return key
    return None


# ---------------------------------------------------------------------------
# Embedding with Ollama
# ---------------------------------------------------------------------------


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings using Ollama and the mxbai-embed-large:335m model.

    This implementation:
      - calls Ollama once per text
      - uses the 'prompt' field (Ollama API)
      - reads the 'embedding' key from the response
    """
    if not texts:
        return []

    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    url = f"{ollama_host.rstrip('/')}/api/embeddings"

    vectors: List[List[float]] = []

    for text in texts:
        payload = {
            "model": settings.embedding_model,  # e.g. "mxbai-embed-large:335m"
            "prompt": text,
        }

        try:
            resp = requests.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Failed request to Ollama at {url}: {e}")

        if "embedding" not in data:
            raise RuntimeError(f"Unexpected response from Ollama: {data}")

        vectors.append(data["embedding"])

    return vectors


# ---------------------------------------------------------------------------
# Chunking + embedding riders
# ---------------------------------------------------------------------------


def build_embedding_text(rider: dict) -> str:
    """
    Build the text we will embed for a rider.
    Uses rider fields + event title when available.
    """
    parts = [
        rider.get("name", ""),
        rider.get("age", ""),
        rider.get("location", ""),
        rider.get("bike", ""),
        rider.get("frame_type", ""),
        rider.get("tyre_width", ""),
        rider.get("key_items", ""),
    ]

    if rider.get("event_title"):
        parts.append(f"Event: {rider['event_title']}")

    return " | ".join(p for p in parts if p)


def chunk_text(
    text: str,
    max_chars: int = 800,
    overlap: int = 100,
) -> List[str]:
    """
    Simple character-based chunking with overlap.
    """
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])

        if end >= n:
            break

        start = max(0, end - overlap)

    return chunks


def embed_rider_chunks(
    rider: dict,
    event_title: str,
    event_url: str,
    event_key: Optional[str],
    max_chars: int = 800,
    overlap: int = 100,
) -> List[dict]:
    """
    Build chunks for a single rider and embed them.

    Returns a list of dicts ready to upsert into Qdrant:
      {
        rider_id, chunk_index, text, vector,
        name, event_title, event_url,
        frame_type, frame_material, wheel_size, tyre_width,
        electronic_shifting, event_key,
      }
    """
    text = build_embedding_text(rider)
    chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)

    if not chunks:
        return []

    vectors = embed_texts(chunks)

    rider_id = rider.get("rider_id")
    name = rider.get("name")
    frame_type = rider.get("frame_type")
    frame_material = rider.get("frame_material")
    wheel_size = rider.get("wheel_size")
    tyre_width = rider.get("tyre_width")
    electronic_shifting = rider.get("electronic_shifting")

    results: List[dict] = []

    for i, (chunk_text_value, vec) in enumerate(zip(chunks, vectors)):
        results.append(
            {
                "rider_id": rider_id,
                "chunk_index": i,
                "text": chunk_text_value,
                "name": name,
                "event_title": event_title,
                "event_url": event_url,
                "frame_type": frame_type,
                "frame_material": frame_material,
                "wheel_size": wheel_size,
                "tyre_width": tyre_width,
                "electronic_shifting": electronic_shifting,
                "event_key": event_key,
                "vector": vec,
            }
        )

    return results


def embed_riders_from_json(
    json_path: Path | str = "data/dotwatcher_bikes_cleaned.json",
    max_chars: int = 800,
    overlap: int = 100,
) -> List[dict]:
    """
    Load DotWatcher articles, flatten riders, and embed all riders.

    Returns a flat list of chunks:
      [{rider_id, chunk_index, text, vector, ...}, ...]
    """
    path = Path(json_path)
    articles = json.loads(path.read_text(encoding="utf-8"))

    all_chunks: List[dict] = []
    rider_global_id = 1

    for article in articles:
        event_title = article.get("title", "")
        event_url = article.get("url", "")
        event_key = infer_event_key_from_title(event_title)

        riders = article.get("riders") or []

        for rider in riders:
            r: Dict = dict(rider)
            r["rider_id"] = rider_global_id
            r["event_title"] = event_title
            r["event_url"] = event_url

            chunks = embed_rider_chunks(
                r,
                event_title=event_title,
                event_url=event_url,
                event_key=event_key,
                max_chars=max_chars,
                overlap=overlap,
            )
            all_chunks.extend(chunks)
            rider_global_id += 1

    return all_chunks
