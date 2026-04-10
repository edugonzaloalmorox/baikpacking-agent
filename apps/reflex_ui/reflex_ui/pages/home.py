"""Home page for the bikepacking chat UI."""

import reflex as rx

from ..components import (
    PAGE_BG,
    MUTED_STYLE,
    TEXT_STYLE,
    bubble,
    key_value_grid,
    metric_tile,
    pill,
    prompt_chip,
    section_heading,
    surface_card,
)
from ..state import BikepackingState, ChatTurn, EXAMPLE_PROMPTS


def _header() -> rx.Component:
    """Render the lightweight product header."""
    return surface_card(
        rx.hstack(
            rx.vstack(
                rx.hstack(
                    rx.box(width="10px", height="10px", border_radius="999px", background_color="#14b8a6"),
                    rx.text("Bikepacking Recommender", font_size="14px", font_weight="700", color="#0f172a"),
                    spacing="2",
                    align="center",
                ),
                rx.text(
                    "A conversational assistant for bikepacking advice, setups, and event-specific recommendations.",
                    **MUTED_STYLE,
                ),
                spacing="1",
                align="start",
            ),
            pill("Chat-first", accent=True),
            justify="between",
            align="center",
            width="100%",
            wrap="wrap",
        ),
        padding="16px",
    )


def _empty_state() -> rx.Component:
    """Render the first-screen welcome state."""
    return surface_card(
        rx.vstack(
            rx.text("What do you need for the next ride?", font_size="22px", font_weight="800", color="#0f172a"),
            rx.text(
                "Ask for a setup, narrow down a component, or follow up on the last answer. The assistant keeps the conversation in one thread.",
                max_width="720px",
                **MUTED_STYLE,
            ),
            rx.hstack(
                *[
                    prompt_chip(prompt, on_click=BikepackingState.load_example(prompt))
                    for prompt in EXAMPLE_PROMPTS
                ],
                spacing="2",
                wrap="wrap",
            ),
            spacing="4",
            align="start",
            width="100%",
        ),
        padding="28px",
    )


def _detail_section(turn: ChatTurn) -> rx.Component:
    """Render the structured details accordion content."""
    return rx.vstack(
        section_heading("Details", "Resolved context", "The technical metadata stays tucked away unless expanded."),
        key_value_grid(
            [
                ("Event", turn.resolved_event_name),
                ("Match type", turn.resolved_event_match_type),
                ("Intent", turn.intent_component),
                ("Policy mode", turn.policy_mode),
                ("Policy notes", turn.policy_notes),
            ]
        ),
        rx.box(
            rx.text(
                "Recommended setup",
                font_size="13px",
                font_weight="700",
                letter_spacing="0.12em",
                text_transform="uppercase",
                color="#64748b",
            ),
            rx.vstack(
                rx.foreach(
                    turn.setup_lines,
                    lambda line: rx.text(line, **TEXT_STYLE),
                ),
                spacing="2",
                align="start",
                width="100%",
            ),
            padding="16px",
            border_radius="18px",
            background_color="#f8fafc",
            border="1px solid rgba(148, 163, 184, 0.12)",
            width="100%",
        ),
        spacing="4",
        align="start",
        width="100%",
    )


def _evidence_section(turn: ChatTurn) -> rx.Component:
    """Render the evidence accordion content."""
    return rx.vstack(
        section_heading("Evidence", "Signals behind the answer", "Helpful when you want to inspect how strong the grounding was."),
        rx.grid(
            metric_tile("Rider count", turn.evidence_rider_count),
            metric_tile("Component hits", turn.evidence_component_hit_count),
            metric_tile("Strength", turn.evidence_strength),
            metric_tile("Consistency", turn.evidence_consistency),
            columns="repeat(auto-fit, minmax(140px, 1fr))",
            gap="10px",
            width="100%",
        ),
        rx.box(
            rx.text(
                "Field support",
                font_size="13px",
                font_weight="700",
                letter_spacing="0.12em",
                text_transform="uppercase",
                color="#64748b",
            ),
            rx.vstack(
                rx.foreach(
                    turn.field_support_lines,
                    lambda line: rx.text(line, **TEXT_STYLE),
                ),
                spacing="2",
                align="start",
                width="100%",
            ),
            padding="16px",
            border_radius="18px",
            background_color="#f8fafc",
            border="1px solid rgba(148, 163, 184, 0.12)",
            width="100%",
        ),
        spacing="4",
        align="start",
        width="100%",
    )


def _why_section(turn: ChatTurn) -> rx.Component:
    """Render the explanation accordion content."""
    return rx.vstack(
        section_heading("Why this recommendation", "Assistant reasoning", "This is the concise explanation the user sees when they expand it."),
        rx.text(turn.reasoning, **TEXT_STYLE, white_space="pre-wrap"),
        spacing="3",
        align="start",
        width="100%",
    )


def _debug_section(turn: ChatTurn) -> rx.Component:
    """Render the optional debug accordion content."""
    return rx.vstack(
        section_heading("Debug", "Backend trace", "Visible only when the backend returned debug data."),
        rx.box(
            rx.text("Retrieval plan", font_size="13px", font_weight="700", color="#0f172a"),
            rx.text(
                turn.retrieval_plan_json,
                font_family="monospace",
                font_size="12px",
                color="#334155",
                white_space="pre-wrap",
            ),
            padding="14px 16px",
            border_radius="16px",
            background_color="#f8fafc",
            border="1px solid rgba(148, 163, 184, 0.12)",
            width="100%",
        ),
        rx.box(
            rx.text("Trace", font_size="13px", font_weight="700", color="#0f172a"),
            rx.text(
                turn.trace_json,
                font_family="monospace",
                font_size="12px",
                color="#334155",
                white_space="pre-wrap",
            ),
            padding="14px 16px",
            border_radius="16px",
            background_color="#f8fafc",
            border="1px solid rgba(148, 163, 184, 0.12)",
            width="100%",
        ),
        spacing="3",
        align="start",
        width="100%",
    )


def _assistant_bubble(turn: ChatTurn) -> rx.Component:
    """Render an assistant turn with progressive disclosure."""
    chips = [
        pill(turn.resolved_event_chip_label, accent=True),
        pill(turn.intent_chip_label),
        pill(turn.policy_chip_label),
    ]

    return bubble(
        "assistant",
        rx.vstack(
            rx.hstack(*chips, spacing="2", wrap="wrap"),
            rx.text(turn.content, **TEXT_STYLE, white_space="pre-wrap"),
            rx.accordion.root(
                rx.accordion.item(
                    header=rx.text("Details", font_size="14px", font_weight="700", color="#0f172a"),
                    content=_detail_section(turn),
                    value="details",
                ),
                rx.accordion.item(
                    header=rx.text("Evidence", font_size="14px", font_weight="700", color="#0f172a"),
                    content=_evidence_section(turn),
                    value="evidence",
                ),
                rx.accordion.item(
                    header=rx.text("Why this recommendation", font_size="14px", font_weight="700", color="#0f172a"),
                    content=_why_section(turn),
                    value="why",
                ),
                type="multiple",
                collapsible=True,
                variant="surface",
                radius="large",
                show_dividers=False,
                width="100%",
            ),
            rx.cond(
                turn.has_debug,
                rx.accordion.root(
                    rx.accordion.item(
                        header=rx.text("Debug", font_size="14px", font_weight="700", color="#0f172a"),
                        content=_debug_section(turn),
                        value="debug",
                    ),
                    type="multiple",
                    collapsible=True,
                    variant="surface",
                    radius="large",
                    show_dividers=False,
                    width="100%",
                ),
                rx.fragment(),
            ),
            spacing="4",
            align="start",
            width="100%",
        ),
        max_width="82%",
    )


def _user_bubble(turn: ChatTurn) -> rx.Component:
    """Render a user turn."""
    return bubble(
        "user",
        rx.vstack(
            rx.text(
                "You",
                font_size="12px",
                font_weight="700",
                letter_spacing="0.12em",
                text_transform="uppercase",
                color="#cbd5e1",
            ),
            rx.text(turn.content, font_size="15px", line_height="1.75", color="inherit", white_space="pre-wrap"),
            spacing="2",
            align="start",
            width="100%",
        ),
        max_width="72%",
    )


def _error_bubble(turn: ChatTurn) -> rx.Component:
    """Render an error turn inside the chat."""
    return bubble(
        "error",
        rx.vstack(
            rx.text(
                "Assistant",
                font_size="12px",
                font_weight="700",
                letter_spacing="0.12em",
                text_transform="uppercase",
                color="inherit",
            ),
            rx.text(turn.content, font_size="15px", line_height="1.75", color="inherit", white_space="pre-wrap"),
            spacing="2",
            align="start",
            width="100%",
        ),
        max_width="78%",
    )


def _render_turn(turn: ChatTurn) -> rx.Component:
    """Render one conversation turn."""
    return rx.box(
        rx.cond(
            turn.role == "user",
            _user_bubble(turn),
            rx.cond(
                turn.role == "error",
                _error_bubble(turn),
                _assistant_bubble(turn),
            ),
        ),
        width="100%",
    )


def _loading_bubble() -> rx.Component:
    """Render the in-flight assistant state."""
    return bubble(
        "assistant",
        rx.hstack(
            rx.spinner(size="3"),
            rx.vstack(
                rx.text(
                    "Bikepacking assistant",
                    font_size="12px",
                    font_weight="700",
                    letter_spacing="0.12em",
                    text_transform="uppercase",
                    color="#64748b",
                ),
                rx.text("Thinking through the recommendation…", font_size="15px", line_height="1.75", color="#0f172a"),
                spacing="1",
                align="start",
            ),
            spacing="3",
            align="center",
        ),
        max_width="72%",
    )


def _composer() -> rx.Component:
    """Render the message composer."""
    return surface_card(
        rx.vstack(
            rx.text(
                "Continue the conversation",
                font_size="13px",
                font_weight="700",
                letter_spacing="0.12em",
                text_transform="uppercase",
                color="#64748b",
            ),
            rx.text_area(
                value=BikepackingState.query,
                on_change=BikepackingState.set_query,
                placeholder="Ask about tyres, bags, full setups, or refine the previous answer…",
                width="100%",
                min_height="110px",
                padding="16px",
                border_radius="18px",
                background_color="#ffffff",
                border="1px solid rgba(148, 163, 184, 0.18)",
                color="#0f172a",
                font_size="16px",
            ),
            rx.hstack(
                rx.text(
                    "You can keep asking follow-up questions in the same thread.",
                    **MUTED_STYLE,
                ),
                rx.spacer(),
                rx.button(
                    rx.cond(
                        BikepackingState.loading,
                        rx.hstack(rx.spinner(size="2"), rx.text("Sending…")),
                        "Send",
                    ),
                    on_click=BikepackingState.submit_query,
                    disabled=rx.cond(BikepackingState.can_send, False, True),
                    background_color="#0f172a",
                    color="#f8fafc",
                    border_radius="14px",
                    padding_x="20px",
                    padding_y="14px",
                    font_weight="700",
                ),
                spacing="3",
                align="center",
                width="100%",
            ),
            spacing="4",
            align="start",
            width="100%",
        ),
        padding="20px",
    )


def _chat_thread() -> rx.Component:
    """Render the scrollable chat transcript."""
    transcript = rx.vstack(
        rx.foreach(BikepackingState.messages, _render_turn),
        rx.cond(BikepackingState.loading, _loading_bubble(), rx.box()),
        spacing="4",
        align="stretch",
        width="100%",
    )

    return surface_card(
        rx.box(
            rx.cond(BikepackingState.has_messages, transcript, _empty_state()),
            max_height="calc(100vh - 330px)",
            overflow_y="auto",
            padding_right="6px",
            width="100%",
        ),
        padding="22px",
    )


@rx.page(route="/", title="Bikepacking Recommender")
def home() -> rx.Component:
    """Render the chat-first bikepacking assistant."""
    return rx.box(
        rx.box(
            position="absolute",
            inset="0",
            background=PAGE_BG,
            z_index="-1",
        ),
        rx.container(
            rx.vstack(
                _header(),
                _chat_thread(),
                _composer(),
                rx.text(
                    "Grounded recommendations powered by the bikepacking backend.",
                    font_size="12px",
                    color="#64748b",
                    text_align="center",
                ),
                spacing="4",
                align="stretch",
                width="100%",
            ),
            max_width="960px",
            padding_x=["14px", "18px", "24px"],
            padding_y=["16px", "22px", "28px"],
            height="100vh",
        ),
        min_height="100vh",
        position="relative",
        overflow="hidden",
    )