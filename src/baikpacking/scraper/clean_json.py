import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------

class Rider(BaseModel):
    """
    Rider schema compatible with pydantic_ai output_type.
    """
    name: Optional[str] = None
    age: Optional[str] = None
    location: Optional[str] = None
    bike: Optional[str] = None
    key_items: Optional[str] = None
    frame_type: Optional[str] = None
    frame_material: Optional[str] = None
    wheel_size: Optional[str] = None
    tyre_width: Optional[str] = None
    electronic_shifting: Optional[bool] = None


# ---------------------------------------------------------------------
# Constants / config
# ---------------------------------------------------------------------

NAV_LINES = {
    "DotWatcher.cc",
    "Event Commentary",
    "Results",
    "Event Calendar",
    "Features",
    "About Us",
    ">",
    "Powered By:",
    "Fastest Times",
    "Bikes of...",
}

CUT_MARKERS = (
    "Also from",
    "Related Features",
    "Proudly Supported By:",
    "The DotWatcher Digest",
    "Privacy Preferences",
    "Submit to DotWatcher",
)

LABEL_TO_FIELD = {
    "age": "age",
    "location": "location",
    "bike": "bike",
    "frame type": "frame_type",
    "frame material": "frame_material",
    "wheel size": "wheel_size",
    "tyre width": "tyre_width",
    "electronic shifting": "electronic_shifting",
}

FIELD_LABELS = set(LABEL_TO_FIELD.keys())
KEY_ITEMS_LABEL = "key items of kit"
CAP_NUMBER_LABEL = "cap number"
MULTILINE_FIELDS = {"bike", "key_items"}
RIDER_FIELD_NAMES = {
    "name",
    "age",
    "location",
    "bike",
    "key_items",
    "frame_type",
    "frame_material",
    "wheel_size",
    "tyre_width",
    "electronic_shifting",
}


# ---------------------------------------------------------------------
# Helpers (pure)
# ---------------------------------------------------------------------

def normalize_label(line: str) -> str:
    """
    Normalize a label line:
    - strip spaces
    - remove trailing colon
    - remove leading 'Your ' (case-insensitive)
    - lower-case
    """
    label = line.strip()
    if label.endswith(":"):
        label = label[:-1].strip()
    if label.lower().startswith("your "):
        label = label[5:].strip()
    return label.lower()


def is_date_line(line: str) -> bool:
    """Return True if the line looks like a date line."""
    return bool(re.match(r"^\d{1,2}\s+\w+\s*,?\s+\d{4}$", line.strip()))


def is_age_label(line: str) -> bool:
    """Return True if this line is an Age label ('Age' or 'Age:')."""
    return normalize_label(line) == "age"


def is_bike_label(line: str) -> bool:
    """Return True if this line is a Bike label ('Bike' or 'Bike:')."""
    return normalize_label(line) == "bike"


def is_known_label(line: str) -> bool:
    """Return True if line is any field/key-items/cap label."""
    norm = normalize_label(line)
    return norm in FIELD_LABELS or norm == KEY_ITEMS_LABEL or norm == CAP_NUMBER_LABEL


def looks_like_name_line(line: str) -> bool:
    """
    Heuristic for rider names.
    Accepts lines like 'Andrew Phillips' but rejects headers and labels.
    """
    text = line.strip()
    if not text:
        return False

    norm = normalize_label(text)

    if text in NAV_LINES:
        return False
    if text in CUT_MARKERS:
        return False
    if text.startswith(">"):
        return False
    if ":" in text:
        return False
    if norm in FIELD_LABELS:
        return False
    if norm in {KEY_ITEMS_LABEL, CAP_NUMBER_LABEL}:
        return False
    if norm == "bikes of":
        return False
    if text.lower().startswith("bikes of "):
        return False
    if is_date_line(text):
        return False
    if re.search(r"\d", text):
        return False
    if len(text.split()) > 5:
        return False

    return True


def _is_noise_line(line: str) -> bool:
    """Return True for navigation/header/footer lines that should be ignored."""
    text = line.strip()
    if not text:
        return True
    if text in NAV_LINES:
        return True
    if text in CUT_MARKERS:
        return True
    if text.startswith(">"):
        return True
    if text.lower().startswith("powered by"):
        return True
    if text.lower().startswith("bikes of") and normalize_label(text) == "bikes of":
        return True
    return False


def _clean_title_candidate(title: str) -> str:
    """Normalize a title candidate by stripping repeated 'Bikes of' prefixes."""
    title = re.sub(r"(?i)^bikes of\s*", "", title).strip()
    title = re.sub(r"\s+", " ", title).strip()
    return title


def extract_title_from_body(raw_body: str) -> Optional[str]:
    """
    Recover article title from raw body when item['title'] is missing or empty.

    Supports:
    - 'Bikes of X'
    - 'Bikes of' followed by 'X' on the next line
    """
    lines = [l.strip() for l in raw_body.splitlines() if l.strip()]
    title_end = len(lines)
    for i, line in enumerate(lines):
        if is_date_line(line):
            title_end = i
            break

    for i, line in enumerate(lines[:title_end]):
        if _is_noise_line(line):
            continue

        if re.fullmatch(r"(?i)bikes of", line):
            if i + 1 < len(lines):
                candidate = lines[i + 1].strip()
                if candidate and not _is_noise_line(candidate) and not is_date_line(candidate):
                    return _clean_title_candidate(candidate)
            continue

        m = re.match(r"(?i)^bikes of\s+(.+)$", line)
        if m:
            candidate = m.group(1).strip()
            if candidate:
                return _clean_title_candidate(candidate)

        if not is_date_line(line) and not normalize_label(line) in FIELD_LABELS:
            return _clean_title_candidate(line)

    return None


def extract_title_from_url(url: Optional[str]) -> Optional[str]:
    """Recover article title from DotWatcher slug."""
    if not url:
        return None

    slug = url.rstrip("/").split("/")[-1]
    slug = re.sub(r"^bikes-of-", "", slug, flags=re.IGNORECASE)
    slug = slug.replace("-", " ").strip()
    if not slug:
        return None
    return _clean_title_candidate(slug.title())


def resolve_article_title(item: dict, raw_body: str, idx: int) -> str:
    """
    Resolve title robustly:
    1. existing title if present
    2. title extracted from body
    3. title derived from url
    4. fallback article_{idx}
    """
    raw_title = (item.get("title") or "").strip()
    if raw_title:
        clean_title = _clean_title_candidate(raw_title)
        if clean_title:
            return clean_title

    body_title = extract_title_from_body(raw_body)
    if body_title:
        return body_title

    url_title = extract_title_from_url(item.get("url"))
    if url_title:
        return url_title

    return f"article_{idx}"


def clean_body(raw_body: str) -> str:
    """
    Remove DotWatcher navigation/header and everything after related-links footer.
    Returns a cleaned body string.
    """
    stripped_lines = [line.strip() for line in raw_body.splitlines() if line.strip()]

    start_idx = 0
    for i, line in enumerate(stripped_lines):
        if _is_noise_line(line):
            continue
        start_idx = i
        break

    cleaned_lines: List[str] = []
    for line in stripped_lines[start_idx:]:
        if _is_noise_line(line):
            continue
        if any(marker in line for marker in CUT_MARKERS):
            break
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def find_name_before_index(lines: List[str], idx: int) -> Optional[str]:
    """
    Walk backwards to find the nearest plausible rider name before a rider anchor.
    """
    for j in range(idx - 1, -1, -1):
        cand = lines[j].strip()
        if not cand:
            continue
        if looks_like_name_line(cand):
            return cand
    return None


def normalize_age(raw: str, article_title: str) -> Optional[str]:
    """
    Validate and normalize the age value.
    Returns a stringified integer or None if invalid.
    """
    raw = raw.strip()
    if not raw:
        return None

    m = re.match(r"(\d{1,3})", raw)
    if not m:
        logger.warning("Invalid age '%s' in article '%s'", raw, article_title)
        return None

    age_int = int(m.group(1))
    if age_int <= 0 or age_int > 120:
        logger.warning("Suspicious age '%s' in article '%s'", raw, article_title)
        return None

    return str(age_int)


def normalize_electronic_shifting(raw: str, article_title: str) -> Optional[bool]:
    """
    Normalize electronic shifting value to True / False / None.
    Accepts various Yes/No variants. Logs when unexpected.
    """
    val = raw.strip().lower()
    mapping_true = {"yes", "y", "true", "sí", "si"}
    mapping_false = {"no", "n", "false"}

    if val in mapping_true:
        return True
    if val in mapping_false:
        return False

    logger.warning(
        "Unexpected electronic_shifting value '%s' in article '%s'",
        raw,
        article_title,
    )
    return None


def _read_inline_or_next_value(lines: List[str], idx: int, label: str) -> tuple[Optional[str], int]:
    """
    Read value for a label, supporting:
    - 'Age: 35'
    - 'Age:' + next line
    - 'Bike:' + next line
    Returns (value, next_index)
    """
    line = lines[idx].strip()

    # inline form
    inline = re.match(rf"(?i)^{re.escape(label)}:?\s*(.+)$", line)
    if inline:
        value = inline.group(1).strip()
        if value and value.lower() != label.lower():
            return value, idx + 1

    # next-line form
    if idx + 1 < len(lines):
        value_line = lines[idx + 1].lstrip(":").strip()
        if (
            value_line
            and not _is_noise_line(value_line)
            and not is_known_label(value_line)
            and not is_date_line(value_line)
        ):
            return value_line, idx + 2

    return None, idx + 1


def is_next_rider_name(lines: List[str], idx: int, lookahead: int = 3) -> bool:
    """
    Return True when the line at idx looks like the start of the next rider block.
    A plausible rider name should be followed shortly by an Age label.
    """
    if idx < 0 or idx >= len(lines):
        return False

    if not looks_like_name_line(lines[idx]):
        return False

    limit = min(len(lines), idx + lookahead + 1)
    for j in range(idx + 1, limit):
        if is_age_label(lines[j]):
            return True

    return False


def parse_riders(cleaned_body: str, article_title: str) -> List[Rider]:
    """
    Parse riders from a cleaned DotWatcher body.

    Strategy:
    - use the line immediately before each Age label as the rider name candidate
    - parse fields only within the block that starts at Age and ends before the next Age
    """
    lines = [l.strip() for l in cleaned_body.splitlines() if l.strip()]
    n = len(lines)
    age_indices = [i for i, line in enumerate(lines) if is_age_label(line)]

    riders: List[Rider] = []

    def _flush_field(rider_data: dict, current_field: Optional[str], buffer: List[str]) -> None:
        if not current_field or not buffer:
            return
        value = "\n".join(v for v in buffer if v).strip()
        if not value:
            return
        if current_field == "electronic_shifting":
            rider_data[current_field] = normalize_electronic_shifting(value, article_title)
        elif current_field == "age":
            rider_data[current_field] = normalize_age(value, article_title)
        elif current_field in MULTILINE_FIELDS:
            rider_data[current_field] = value
        else:
            rider_data[current_field] = value

    for pos, age_idx in enumerate(age_indices):
        if age_idx == 0:
            continue

        name_candidate = lines[age_idx - 1].strip()
        if not looks_like_name_line(name_candidate):
            continue

        next_age_idx = age_indices[pos + 1] if pos + 1 < len(age_indices) else n
        rider_data: dict = {"name": name_candidate}
        current_field: Optional[str] = None
        current_buffer: List[str] = []

        age_value, cursor = _read_inline_or_next_value(lines, age_idx, "Age")
        if age_value:
            rider_data["age"] = normalize_age(age_value, article_title)

        j = cursor
        while j < next_age_idx:
            line = lines[j].strip()
            if not line:
                j += 1
                continue

            label_norm = normalize_label(line)

            if label_norm == KEY_ITEMS_LABEL:
                _flush_field(rider_data, current_field, current_buffer)
                current_field = "key_items"
                current_buffer = []
                value, j2 = _read_inline_or_next_value(lines, j, "Key items of kit")
                if value and not is_next_rider_name(lines, j + 1):
                    current_buffer.append(value)
                j = j2
                continue

            if label_norm == CAP_NUMBER_LABEL:
                _flush_field(rider_data, current_field, current_buffer)
                current_field = None
                current_buffer = []
                _, j = _read_inline_or_next_value(lines, j, "Cap number")
                continue

            if label_norm in FIELD_LABELS:
                _flush_field(rider_data, current_field, current_buffer)
                current_field = None
                current_buffer = []

                field_name = LABEL_TO_FIELD[label_norm]
                value_raw, j2 = _read_inline_or_next_value(lines, j, line)

                if value_raw:
                    if field_name == "electronic_shifting":
                        rider_data[field_name] = normalize_electronic_shifting(
                            value_raw,
                            article_title,
                        )
                    elif field_name == "age":
                        rider_data[field_name] = normalize_age(value_raw, article_title)
                    else:
                        rider_data[field_name] = value_raw

                    if field_name in MULTILINE_FIELDS:
                        current_field = field_name
                        current_buffer = []
                        if not is_next_rider_name(lines, j + 1):
                            current_buffer = [value_raw]

                j = j2
                continue

            if (
                current_field in MULTILINE_FIELDS
                and not _is_noise_line(line)
                and not is_next_rider_name(lines, j)
            ):
                current_buffer.append(line.lstrip(":").strip())
                j += 1
                continue

            j += 1

        _flush_field(rider_data, current_field, current_buffer)

        if any(rider_data.get(field) not in (None, "") for field in RIDER_FIELD_NAMES):
            riders.append(Rider(**rider_data))

    return riders


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_articles(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return list(_iter_jsonl(path))

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data

    raise ValueError("Unexpected input format: expected list/dict JSON or JSONL rows.")


def _latest_raw_new_snapshot(raw_snap_dir: Path) -> Path:
    files = sorted(
        list(raw_snap_dir.glob("dotwatcher_bikes_raw_new_*.json"))
        + list(raw_snap_dir.glob("dotwatcher_bikes_raw_new_*.jsonl"))
    )
    if not files:
        raise FileNotFoundError(
            f"No raw new-only snapshots found in {raw_snap_dir}. "
            f"Expected dotwatcher_bikes_raw_new_*.json or .jsonl"
        )
    return files[-1]


def _extract_run_id_from_raw_snapshot(path: Path) -> str:
    """
    Extract run id from raw snapshot path:
    dotwatcher_bikes_raw_new_<RUN_ID>.jsonl
    """
    name = path.name
    prefix = "dotwatcher_bikes_raw_new_"
    if prefix in name:
        return name.split(prefix, 1)[1].split(".", 1)[0]
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Clean DotWatcher raw snapshots into cleaned snapshots."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Path to raw snapshot (.json or .jsonl). If omitted, uses latest raw new-only snapshot in data/snapshots/raw/.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output path for cleaned snapshot (.json). If omitted, writes to data/snapshots/clean/ with matching run_id.",
    )
    parser.add_argument(
        "--update-latest",
        action="store_true",
        help="Also update data/dotwatcher_bikes_cleaned.json by merging new cleaned rows.",
    )
    args = parser.parse_args()

    raw_snap_dir = Path("data/snapshots/raw")
    clean_snap_dir = Path("data/snapshots/clean")
    clean_snap_dir.mkdir(parents=True, exist_ok=True)

    in_path = Path(args.input) if args.input else _latest_raw_new_snapshot(raw_snap_dir)
    run_id = _extract_run_id_from_raw_snapshot(in_path)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = clean_snap_dir / f"dotwatcher_bikes_cleaned_new_{run_id}.json"

    logger.info("Loading raw articles from %s", in_path)
    articles = _load_articles(in_path)
    logger.info("Processing %d articles", len(articles))

    for idx, item in enumerate(articles):
        raw_body = item.get("body", "")

        title = resolve_article_title(item, raw_body, idx)
        item["title"] = title

        cleaned = clean_body(raw_body)
        item["body"] = cleaned

        riders = parse_riders(cleaned, title)
        item["riders"] = [r.model_dump() for r in riders]

        logger.debug(
            "Article %d ('%s'): parsed %d riders",
            idx,
            title,
            len(riders),
        )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    logger.info("Saved cleaned new-only snapshot to %s", out_path)

    if args.update_latest:
        latest_path = Path("data/dotwatcher_bikes_cleaned.json")
        existing_urls = set()
        merged = []

        if latest_path.exists():
            with latest_path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                merged.extend(existing)
                for row in existing:
                    url = row.get("url")
                    if url:
                        existing_urls.add(url)

        new_added = 0
        for row in articles:
            url = row.get("url")
            if not url or url in existing_urls:
                continue
            merged.append(row)
            existing_urls.add(url)
            new_added += 1

        latest_path.parent.mkdir(parents=True, exist_ok=True)
        with latest_path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        logger.info(
            "Updated latest cleaned file %s (added %d new rows)",
            latest_path,
            new_added,
        )


if __name__ == "__main__":
    main()
