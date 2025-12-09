from typing import Optional, List, Dict, Any
import requests
import unicodedata

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from .config import Settings
from .embed import embed_texts


settings = Settings()


# ---------------------------------------------------------------------------
# Qdrant client + collection management
# ---------------------------------------------------------------------------


def get_qdrant_client() -> QdrantClient:
    """
    Build a QdrantClient from settings.
    """
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=60.0,
    )


def ensure_collection(vector_size: int, client: Optional[QdrantClient] = None) -> QdrantClient:
    """
    Ensure the target collection exists with the correct vector size.

    - If it exists: do nothing.
    - If it doesn't: create it with COSINE distance.
    """
    client = client or get_qdrant_client()
    name = settings.qdrant_collection

    collections = client.get_collections().collections
    if any(c.name == name for c in collections):
        return client

    client.create_collection(
        collection_name=name,
        vectors_config=rest.VectorParams(
            size=vector_size,
            distance=rest.Distance.COSINE,
        ),
    )
    return client


def upsert_chunks_to_qdrant(chunks: List[Dict], batch_size: int = 500) -> None:
    """
    Upsert rider chunks into Qdrant in batches.

    Each chunk dict is expected to have:
      - rider_id
      - chunk_index
      - text
      - vector
      - optionally other metadata (name, event_title, event_url, frame_type, ...)
      - optionally event_key (used for event-aware search)
    """
    if not chunks:
        print("No chunks to upsert into Qdrant.")
        return

    client = get_qdrant_client()
    vec_size = len(chunks[0]["vector"])
    ensure_collection(vec_size, client=client)

    collection_name = settings.qdrant_collection

    total = len(chunks)
    print(f"Upserting {total} chunks into Qdrant (batch_size={batch_size})...")

    global_point_id = 1  # integer IDs as Qdrant expects

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = chunks[start:end]

        points: List[rest.PointStruct] = []

        for chunk in batch:
            point_id = global_point_id
            global_point_id += 1

            payload = {k: v for k, v in chunk.items() if k != "vector"}

            points.append(
                rest.PointStruct(
                    id=point_id,
                    vector=chunk["vector"],
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

        print(f"  Upserted {end} / {total} chunks")

    print(f"Finished upserting {total} chunks into collection '{collection_name}'.")
    

# ---------------------------------------------------------------------------
# Search helpers (semantic + event-aware)
# ---------------------------------------------------------------------------


def search_riders(
    query: str,
    top_k: int = 5,
    event_key: Optional[str] = None,
) -> List[Dict]:
    """
    Semantic search over rider chunks in Qdrant, using direct HTTP to the
    /collections/{collection_name}/points/search endpoint.

    If event_key is provided, we filter on payload.event_key == event_key, so
    only riders from that race/event are considered.
    """
    # 1. Embed the query with Ollama
    vectors = embed_texts([query])
    if not vectors:
        return []

    query_vector = vectors[0]

    # 2. Build HTTP request to Qdrant
    base_url = settings.qdrant_url.rstrip("/")
    collection_name = settings.qdrant_collection
    url = f"{base_url}/collections/{collection_name}/points/search"

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if settings.qdrant_api_key:
        headers["api-key"] = settings.qdrant_api_key

    payload: Dict[str, Any] = {
        "vector": query_vector,
        "limit": top_k,
        "with_payload": True,
        "with_vectors": False,
    }

    # Optional event filter: restrict search to a specific race/event
    if event_key:
        payload["filter"] = {
            "must": [
                {
                    "key": "event_key",
                    "match": {"value": event_key},
                }
            ]
        }

    resp = requests.post(url, json=payload, headers=headers, timeout=30)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Qdrant search failed: {resp.status_code} {resp.text}"
        )

    data = resp.json()
    results = data.get("result") or []

    hits: List[Dict] = []
    for r in results:
        hits.append(
            {
                "id": r.get("id"),
                "score": r.get("score"),
                "payload": r.get("payload") or {},
            }
        )

    return hits


def group_hits_by_rider(
    hits: List[Dict],
    top_k_riders: int = 5,
    max_chunks_per_rider: int = 3,
) -> List[Dict[str, Any]]:
    """
    Group raw Qdrant hits (chunk-level) by rider_id and return a ranked
    list of riders.

    Each returned item has:
      - rider_id
      - best_score
      - name
      - event_title
      - event_url
      - frame_type
      - frame_material
      - wheel_size
      - tyre_width
      - electronic_shifting
      - event_key
      - chunks: list of {score, text, chunk_index}
    """
    by_rider: Dict[Any, Dict[str, Any]] = {}

    for hit in hits:
        payload = hit.get("payload") or {}
        rider_id = payload.get("rider_id")

        if rider_id is None:
            continue

        score = hit.get("score", 0.0)
        name = payload.get("name")
        event_title = payload.get("event_title")
        event_url = payload.get("event_url")
        frame_type = payload.get("frame_type")
        frame_material = payload.get("frame_material")
        wheel_size = payload.get("wheel_size")
        tyre_width = payload.get("tyre_width")
        electronic_shifting = payload.get("electronic_shifting")
        event_key = payload.get("event_key")

        text = payload.get("text", "")
        chunk_index = payload.get("chunk_index", 0)

        if rider_id not in by_rider:
            by_rider[rider_id] = {
                "rider_id": rider_id,
                "name": name,
                "event_title": event_title,
                "event_url": event_url,
                "frame_type": frame_type,
                "frame_material": frame_material,
                "wheel_size": wheel_size,
                "tyre_width": tyre_width,
                "electronic_shifting": electronic_shifting,
                "event_key": event_key,
                "best_score": score,
                "chunks": [],
            }

        agg = by_rider[rider_id]

        if score > agg["best_score"]:
            agg["best_score"] = score

        agg["chunks"].append(
            {
                "score": score,
                "text": text,
                "chunk_index": chunk_index,
            }
        )

    riders = list(by_rider.values())

    # Sort chunks per rider by score (desc) and truncate
    for r in riders:
        r["chunks"].sort(key=lambda c: c["score"], reverse=True)
        r["chunks"] = r["chunks"][:max_chunks_per_rider]

    # Sort riders by best_score (desc) and truncate
    riders.sort(key=lambda r: r["best_score"], reverse=True)
    return riders[:top_k_riders]


# ---------------------------------------------------------------------------
# Event detection / normalisation
# ---------------------------------------------------------------------------


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


def infer_event_key_from_title(title: str) -> Optional[str]:
    """
    Infer a canonical event_key from a DotWatcher article title (used at
    embedding time when building payloads).
    """
    norm = _normalize_text_for_match(title)
    for key, aliases in EVENT_ALIASES.items():
        for alias in aliases:
            if alias in norm:
                return key
    return None


def _detect_event_key_in_query(query: str) -> Optional[str]:
    """
    Try to detect a canonical event key in the query using EVENT_ALIASES.
    """
    q_norm = _normalize_text_for_match(query)
    if not q_norm:
        return None

    for key, aliases in EVENT_ALIASES.items():
        for alias in aliases:
            if alias in q_norm:
                return key
    return None


# ---------------------------------------------------------------------------
# High-level grouped search (used by the recommender agent)
# ---------------------------------------------------------------------------


def search_riders_grouped(
    query: str,
    top_k_riders: int = 10,
    oversample_factor: int = 10,
    max_chunks_per_rider: int = 3,
) -> List[Dict[str, Any]]:
    """
    High-level helper:

      1) Detect event key in the query (e.g. 'transcontinental', 'transiberica').
      2) If an event_key is found, search ONLY within that event using a
         Qdrant filter on payload.event_key.
      3) Otherwise, do a generic global semantic search.
      4) Group results by rider_id and return the top_k_riders.
    """
    event_key = _detect_event_key_in_query(query)

    raw_hits = search_riders(
        query,
        top_k=top_k_riders * oversample_factor,
        event_key=event_key,
    )

    if not raw_hits:
        return []

    riders = group_hits_by_rider(
        raw_hits,
        top_k_riders=top_k_riders * oversample_factor,
        max_chunks_per_rider=max_chunks_per_rider,
    )

    return riders[:top_k_riders]
