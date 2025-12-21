

import ast
import json
import os
import re
from typing import Any, Union

from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class EvaluationResult(BaseModel):
    meets_constraints: bool = False
    covers_required_components: bool = False
    includes_required_items: bool = False
    avoids_forbidden_items: bool = False
    does_not_invent_event_facts: bool = False
    output_is_actionable: bool = False

    score_0_to_6: int = Field(ge=0, le=6)
    notes: str = ""
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a strict offline evaluator (LLM-as-a-judge).
You MUST judge ONLY using the provided QUESTION, INSTRUCTIONS, GROUND_TRUTH, ANSWER and MESSAGES.
You may NOT use external knowledge.

Return ONLY valid JSON matching the schema. No markdown. No extra keys.
If the ground truth does not contain enough information to verify something, mark that criterion false and explain why in notes.
"""

JUDGE_PROMPT_TEMPLATE = """
Evaluate the candidate answer (<ANSWER>) to the user question (<QUESTION>).

Use ONLY:
- <INSTRUCTIONS>
- <GROUND_TRUTH>
- <ANSWER>
- <MESSAGES>

Do NOT use external knowledge. Do NOT assume facts not present in ground truth.

Checklist criteria:
- meets_constraints: respects explicit constraints in the QUESTION (e.g., avoid electronic shifting, prioritize waterproof packing, minimal kit)
- covers_required_components: covers every item in must_cover from the ground truth
- includes_required_items: mentions every item in must_include from the ground truth
- avoids_forbidden_items: does not recommend items listed in must_avoid from the ground truth
- does_not_invent_event_facts: does not assert event facts (route, distance, mandatory kit, rules) unless they appear in GROUND_TRUTH or QUESTION
- output_is_actionable: provides a usable recommendation with concrete choices/tradeoffs

Scoring:
score_0_to_6 = number of checklist items set to true (6 max).

Scoring:
score_0_to_5 = number of checklist items set to true (5 max).

<QUESTION>
{question}
</QUESTION>

<INSTRUCTIONS>
{instructions}
</INSTRUCTIONS>

<GROUND_TRUTH>
{ground_truth}
</GROUND_TRUTH>

<ANSWER>
{answer}
</ANSWER>

<MESSAGES>
{messages_json}
</MESSAGES>


Output JSON keys MUST be exactly:
meets_constraints, covers_required_components, includes_required_items, avoids_forbidden_items,
does_not_invent_event_facts, output_is_actionable, score_0_to_6, notes, tags

Do NOT output any other keys like notes_meets_constraints or score.

"""


# ---------------------------------------------------------------------------
# Regex / parsing helpers
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)

# Matches: AgentRunResult(output='...') and extracts the quoted body.
_OUTPUT_RE = re.compile(r"output=(?P<q>['\"])(?P<body>.*?)(?P=q)\)\s*$", re.DOTALL)

# Fallback: first JSON object found anywhere in text.
_JSON_OBJ_RE = re.compile(r"(\{.*\})", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove <think> blocks if a model outputs them."""
    return _THINK_RE.sub("", text).strip()


def dumps_messages(messages: Any) -> str:
    """Serialize messages/logs into JSON for the prompt."""
    try:
        return json.dumps(messages, ensure_ascii=False, indent=2)
    except TypeError:
        return json.dumps(str(messages), ensure_ascii=False)


def unwrap_agent_output(text: str) -> str:
    """
    Convert wrappers to the raw inner string.

    Handles:
      - already-pure JSON: "{...}"
      - AgentRunResult(output='...'): returns the inner string with escapes decoded
      - fallback: extracts the first {...} substring if present
    """
    s = text.strip()

    if s.startswith("{") and s.endswith("}"):
        return s

    m = _OUTPUT_RE.search(s)
    if m:
        literal = m.group("q") + m.group("body") + m.group("q")
        return ast.literal_eval(literal).strip()

    m2 = _JSON_OBJ_RE.search(s)
    if m2:
        return m2.group(1).strip()

    return s


def extract_text_from_run_result(res: Any) -> str:
    """
    pydantic_ai return types vary by version. This tries a few common shapes.

    Preference order:
      - res.output (if present and str)
      - res.data (if str)
      - res.data.output_text / text / content (if present and str)
      - stringified fallback
    """
    # Best case: direct .output
    out = getattr(res, "output", None)
    if isinstance(out, str):
        return out

    raw = getattr(res, "data", res)

    if isinstance(raw, str):
        return raw

    for attr in ("output_text", "text", "content"):
        val = getattr(raw, attr, None)
        if isinstance(val, str):
            return val

    return str(raw)


def parse_evaluation_result(raw_text: str) -> EvaluationResult:
    text = strip_think(raw_text)
    json_text = unwrap_agent_output(text)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Judge did not return valid JSON.\nRAW:\n{raw_text}\n\nJSON_TEXT:\n{json_text}"
        ) from e


    # 1) score key: accept "score" or "score_0_to_6"
    if "score_0_to_6" not in data:
        if "score" in data and isinstance(data["score"], int):
            data["score_0_to_6"] = data["score"]
        elif "score_0_to_5" in data and isinstance(data["score_0_to_5"], int):
            # if you ever have old outputs
            data["score_0_to_6"] = data["score_0_to_5"]

    # 2) Merge per-criterion notes_* into a single notes string
    note_keys = sorted([k for k in data.keys() if k.startswith("notes_") and k != "notes"])
    if note_keys:
        merged = "\n".join(f"- {k.replace('notes_', '')}: {data[k]}" for k in note_keys if data.get(k))
        base_notes = data.get("notes") if isinstance(data.get("notes"), str) else ""
        data["notes"] = (base_notes + ("\n" if base_notes and merged else "") + merged).strip()

        # optional: remove the extra keys so they don't confuse you later
        for k in note_keys:
            data.pop(k, None)

    # 3) Ensure notes is always a string
    if data.get("notes") is None:
        data["notes"] = ""
    elif not isinstance(data["notes"], str):
        data["notes"] = str(data["notes"])

    # 4) Safety: clamp score range if model returns nonsense
    if isinstance(data.get("score_0_to_6"), int):
        data["score_0_to_6"] = max(0, min(6, data["score_0_to_6"]))
    
    tags = data.get("tags")
    if tags is None:
        data["tags"] = []
    elif isinstance(tags, str):
        # allow "", "tag1, tag2", or single tag
        s = tags.strip()
        if not s:
            data["tags"] = []
        elif "," in s:
            data["tags"] = [t.strip() for t in s.split(",") if t.strip()]
        else:
            data["tags"] = [s]
    elif isinstance(tags, list):
        # ensure list[str]
        data["tags"] = [str(t).strip() for t in tags if str(t).strip()]
    else:
        data["tags"] = [str(tags).strip()] if str(tags).strip() else []

    try:
        return EvaluationResult.model_validate(data)
    except ValidationError as e:
        raise RuntimeError(
            f"Judge returned JSON but it didn't match schema.\nJSON:\n{data}"
        ) from e




# ---------------------------------------------------------------------------
# Model / agent wiring
# ---------------------------------------------------------------------------

def build_judge_model() -> OpenAIChatModel:
    """
    Build a judge model.

    Env vars:
      - JUDGE_MODEL (default: deepseek-r1:8b)
      - OLLAMA_BASE_URL (default: http://localhost:11434/v1)
    """
    model_name = os.getenv("JUDGE_MODEL", "deepseek-r1:8b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    provider = OllamaProvider(base_url=base_url)
    return OpenAIChatModel(model_name, provider=provider)


def build_judge_agent() -> Agent[None, str]:
    """
    Tool-less judge agent (no output_type), returning raw text.
    We parse JSON ourselves to avoid tool/structured-output mechanisms.
    """
    model = build_judge_model()
    return Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        model_settings={"temperature": 0.0},
    )


def build_prompt(
    *,
    question: str,
    answer: str,
    ground_truth: str,
    messages_json: str,
    instructions: str = "",
) -> str:
    """Build the judge prompt."""
    return JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        instructions=instructions or "",
        ground_truth=ground_truth or "",
        answer=answer or "",
        messages_json=messages_json,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def judge_one(
    *,
    question: str,
    answer: str,
    ground_truth: str,
    messages: Any,
    instructions: str = "",
) -> EvaluationResult:
    """
    Evaluate a single row.

    Args:
      question: user query
      answer: candidate answer from the agent
      ground_truth: reference answer / expected fields / rubric text
      messages: full agent trace (list/dict/string)
      instructions: optional user instructions for that row

    Returns:
      EvaluationResult
    """
    agent = build_judge_agent()
    messages_json = dumps_messages(messages)

    prompt = build_prompt(
        question=question,
        answer=answer,
        ground_truth=ground_truth,
        messages_json=messages_json,
        instructions=instructions,
    )

    res = await agent.run(prompt)
    raw_text = extract_text_from_run_result(res)
    return parse_evaluation_result(raw_text)
