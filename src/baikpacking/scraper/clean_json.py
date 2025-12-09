import json
import logging
import re
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
}

CUT_MARKER = "Also from"

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
    """Return True if the line looks like a date line: '24 November, 2025'."""
    return bool(re.match(r"^\d{1,2}\s+\w+,\s+\d{4}$", line))


def is_age_label(line: str) -> bool:
    """Return True if this line is an Age label ('Age' or 'Age:')."""
    return normalize_label(line) == "age"


def clean_body(raw_body: str) -> str:
    """
    Remove DotWatcher navigation header and everything after CUT_MARKER.
    Returns a cleaned body string.
    """
    # 1) Remove nav/header at the top
    raw_lines = raw_body.splitlines()
    cleaned_lines: List[str] = []
    dropping_header = True

    for line in raw_lines:
        stripped = line.strip()
        if dropping_header and (stripped == "" or stripped in NAV_LINES):
            # still skipping leading nav / blank lines
            continue
        else:
            dropping_header = False
            cleaned_lines.append(line)

    body = "\n".join(cleaned_lines)

    # 2) Cut after CUT_MARKER (e.g. "Also from")
    if CUT_MARKER in body:
        body = body.split(CUT_MARKER, 1)[0].rstrip()

    return body


def find_name_for_age(lines: List[str], age_idx: int) -> Optional[str]:
    """
    Given the index of an Age line, walk backwards to find the rider's name.

    We skip:
    - title lines (starting with 'Bikes of ')
    - date lines
    - field labels
    - cap number
    - key items label
    """
    for j in range(age_idx - 1, -1, -1):
        cand = lines[j].strip()
        if not cand:
            continue

        norm = normalize_label(cand)

        if norm in FIELD_LABELS:
            continue
        if norm == KEY_ITEMS_LABEL:
            continue
        if norm == CAP_NUMBER_LABEL:
            continue
        if cand.startswith("Bikes of "):
            continue
        if is_date_line(cand):
            continue

        # looks like a reasonable name line
        return cand

    logger.debug("No name found for Age at index %d", age_idx)
    return None


def normalize_age(raw: str, article_title: str) -> Optional[str]:
    """
    Validate and normalize the age value.
    Returns a stringified integer or None if invalid.
    """
    raw = raw.strip()
    if not raw:
        return None

    # allow "47" or "47 years" etc.
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


def parse_riders(cleaned_body: str, article_title: str) -> List[Rider]:
    """
    Parse riders from a CLEANED DotWatcher body.
    We treat each Age block as the start of a rider.

    Supports:
    - 'Age'
    - 'Age:'
    - 'Age: 35'
    - 'Age:\\n35'
    """
    raw_lines = cleaned_body.splitlines()
    # drop empty lines and strip
    lines = [l.strip() for l in raw_lines if l.strip()]
    n = len(lines)

    riders: List[Rider] = []
    i = 0

    while i < n:
        line = lines[i]

        if is_age_label(line):
            # New rider anchored at this Age / Age:
            name = find_name_for_age(lines, i)
            rider_data: dict = {"name": name}

            # ---- Age value: inline or on next line ----
            age_val_raw = None

            # Case 1: "Age: 35"
            m_inline = re.search(r"Age:?\s*(\d{1,3})", line, flags=re.IGNORECASE)
            if m_inline:
                age_val_raw = m_inline.group(1)
            # Case 2: "Age:" on one line, "35" on the next
            elif i + 1 < n:
                next_line = lines[i + 1].lstrip(":").strip()
                if re.match(r"^\d{1,3}$", next_line):
                    age_val_raw = next_line

            if age_val_raw:
                rider_data["age"] = normalize_age(age_val_raw, article_title)

            # Scan forward until the next Age (any form) or end
            j = i + 1
            key_items_buffer: List[str] = []

            while j < n and not is_age_label(lines[j]):
                l = lines[j]
                label_norm = normalize_label(l)

                # ---- Key items block ----
                if label_norm == KEY_ITEMS_LABEL:
                    key_items_buffer = []

                    # next line might be ": ..." or directly first item
                    j += 1
                    if j < n:
                        first_line = lines[j].lstrip(":").strip()
                        if first_line:
                            key_items_buffer.append(first_line)
                            j += 1

                    # collect until we hit a boundary
                    while j < n:
                        l2 = lines[j]
                        label_norm2 = normalize_label(l2)
                        if (
                            label_norm2 in FIELD_LABELS
                            or label_norm2 == CAP_NUMBER_LABEL
                            or label_norm2 == KEY_ITEMS_LABEL
                            or is_age_label(l2)
                            or l2.startswith("Bikes of ")
                            or is_date_line(l2)
                        ):
                            break
                        key_items_buffer.append(l2)
                        j += 1

                    if key_items_buffer:
                        rider_data["key_items"] = "\n".join(key_items_buffer)
                    continue

                # ---- Ignore Cap number / Cap number: ----
                if label_norm == CAP_NUMBER_LABEL:
                    # value is on next line like ": 33" or "33"
                    if j + 1 < n:
                        j += 2
                    else:
                        j += 1
                    continue

                # ---- Simple field labels (Location, Bike, Frame type, etc.) ----
                if label_norm in FIELD_LABELS:
                    field_name = LABEL_TO_FIELD[label_norm]

                    value_raw = None
                    if j + 1 < n:
                        value_line = lines[j + 1].lstrip(":").strip()
                        if value_line:
                            value_raw = value_line

                    if value_raw:
                        if field_name == "electronic_shifting":
                            rider_data[field_name] = normalize_electronic_shifting(
                                value_raw, article_title
                            )
                        elif field_name == "age":
                            rider_data[field_name] = normalize_age(
                                value_raw, article_title
                            )
                        else:
                            rider_data[field_name] = value_raw

                    if j + 1 < n:
                        j += 2
                    else:
                        j += 1
                    continue

                # otherwise just move on
                j += 1

            # close this rider
            riders.append(Rider(**rider_data))
            i = j  # continue from where we stopped (either next Age or end)
        else:
            i += 1

    return riders


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    in_path = Path("data/dotwatcher_bikes_raw.json")
    out_path = Path("data/dotwatcher_bikes_cleaned.json")

    logger.info("Loading raw articles from %s", in_path)
    with in_path.open("r", encoding="utf-8") as f:
        articles = json.load(f)

    logger.info("Processing %d articles", len(articles))
    
    for idx, item in enumerate(articles):
        raw_body = item.get("body", "")
        title = item.get("title", f"article_{idx}")
        
        title = re.sub(r"(?i)bikes of", "", title).strip()

        articles[idx]["title"] = title
        
        cleaned = clean_body(raw_body)
        item["body"] = cleaned  # keep cleaned body in final JSON

        riders = parse_riders(cleaned, title)
        item["riders"] = [r.model_dump() for r in riders]

        logger.debug(
            "Article %d ('%s'): parsed %d riders",
            idx,
            title,
            len(riders),
        )

    # quick sanity log on first article
    if articles:
        sample_riders = articles[0].get("riders", [])[:3]
        logger.info("Sample riders from first article: %s", sample_riders)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    logger.info("Saved cleaned data to %s", out_path)


if __name__ == "__main__":
    main()
