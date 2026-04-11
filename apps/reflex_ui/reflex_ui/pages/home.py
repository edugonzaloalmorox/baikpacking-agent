"""Home page for the bikepacking chat UI."""

import reflex as rx

from ..components import (
    PAGE_BG,
    MUTED_STYLE,
    TEXT_STYLE,
    bubble,
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
                    rx.box(
                        width="10px",
                        height="10px",
                        border_radius="999px",
                        background_color="#14b8a6",
                    ),
                    rx.text(
                        "Bikepacking Advisor",
                        font_size="14px",
                        font_weight="700",
                        color="#0f172a",
                    ),
                    spacing="2",
                    align="center",
                ),
                rx.text(
                    "Ask for a setup or a component recommendation for a bikepacking event.",
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
            rx.text(
                "Ask for a recommendation",
                font_size="24px",
                font_weight="800",
                color="#0f172a",
            ),
            rx.text(
                "Describe the event or ask about a specific component like tyres, bags, drivetrain, wheels, or bike choice.",
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


def _why_section(turn: ChatTurn) -> rx.Component:
    """Render the explanation section."""
    return rx.vstack(
        section_heading(
            "Why this recommendation",
            "Assistant reasoning",
            "A concise explanation of the recommendation.",
        ),
        rx.text(turn.reasoning, **TEXT_STYLE, white_space="pre-wrap"),
        spacing="3",
        align="start",
        width="100%",
    )


def _feedback_section(turn: ChatTurn) -> rx.Component:
    """Render inline feedback controls for a recommendation turn."""
    controls = rx.hstack(
        rx.button(
            "👍",
            on_click=BikepackingState.submit_feedback(turn.run_id, "thumbs_up"),
            disabled=rx.cond(turn.feedback_status != "", True, False),
            variant="soft",
            color_scheme="green",
            border_radius="999px",
            padding_x="14px",
            padding_y="10px",
            font_weight="700",
        ),
        rx.button(
            "👎",
            on_click=BikepackingState.open_feedback_form(turn.run_id),
            disabled=rx.cond(turn.feedback_status != "", True, False),
            variant="soft",
            color_scheme="red",
            border_radius="999px",
            padding_x="14px",
            padding_y="10px",
            font_weight="700",
        ),
        spacing="2",
        wrap="wrap",
    )

    comment_form = rx.cond(
        turn.feedback_form_open,
        rx.vstack(
            rx.text_area(
                value=turn.feedback_comment,
                on_change=BikepackingState.set_feedback_comment(turn.run_id),
                placeholder="Optional note about what was wrong with the recommendation…",
                width="100%",
                min_height="88px",
                padding="12px",
                border_radius="14px",
                background_color="#ffffff",
                border="1px solid rgba(148, 163, 184, 0.18)",
                color="#0f172a",
                font_size="14px",
            ),
            rx.hstack(
                rx.text(
                    "Add a short comment, or send feedback without one.",
                    **MUTED_STYLE,
                ),
                rx.spacer(),
                rx.button(
                    "Send feedback",
                    on_click=BikepackingState.submit_feedback(turn.run_id, "thumbs_down"),
                    disabled=rx.cond(turn.feedback_status != "", True, False),
                    background_color="#0f172a",
                    color="#f8fafc",
                    border_radius="12px",
                    padding_x="16px",
                    padding_y="10px",
                    font_weight="700",
                ),
                spacing="3",
                align="center",
                width="100%",
            ),
            spacing="3",
            align="start",
            width="100%",
        ),
        rx.fragment(),
    )

    status_line = rx.cond(
        turn.feedback_status != "",
        rx.text(
            f"Feedback recorded: {turn.feedback_status.replace('_', ' ')}",
            font_size="13px",
            color="#475569",
            font_weight="600",
        ),
        rx.fragment(),
    )

    error_line = rx.cond(
        turn.feedback_error != "",
        rx.text(
            turn.feedback_error,
            font_size="13px",
            color="#b91c1c",
            font_weight="600",
        ),
        rx.fragment(),
    )

    return rx.vstack(
        rx.text(
            "Feedback",
            font_size="13px",
            font_weight="700",
            letter_spacing="0.12em",
            text_transform="uppercase",
            color="#64748b",
        ),
        controls,
        comment_form,
        status_line,
        error_line,
        spacing="3",
        align="start",
        width="100%",
    )


def _assistant_bubble(turn: ChatTurn) -> rx.Component:
    """Render an assistant turn with a compact recommendation-first layout."""
    chips = [
        pill(turn.resolved_event_chip_label, accent=True),
        pill(turn.intent_chip_label),
        pill(turn.policy_chip_label),
    ]

    return bubble(
        "assistant",
        rx.vstack(
            rx.hstack(*chips, spacing="2", wrap="wrap"),
            rx.box(
                rx.text(
                    "Recommendation",
                    font_size="13px",
                    font_weight="700",
                    letter_spacing="0.12em",
                    text_transform="uppercase",
                    color="#64748b",
                ),
                rx.text(turn.content, **TEXT_STYLE, white_space="pre-wrap"),
                padding="16px",
                border_radius="18px",
                background_color="#f8fafc",
                border="1px solid rgba(148, 163, 184, 0.12)",
                width="100%",
            ),
            rx.accordion.root(
                rx.accordion.item(
                    header=rx.text(
                        "Why this recommendation",
                        font_size="14px",
                        font_weight="700",
                        color="#0f172a",
                    ),
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
            _feedback_section(turn),
            spacing="4",
            align="start",
            width="100%",
        ),
        max_width="88%",
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
            rx.text(
                turn.content,
                font_size="15px",
                line_height="1.75",
                color="inherit",
                white_space="pre-wrap",
            ),
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
            rx.text(
                turn.content,
                font_size="15px",
                line_height="1.75",
                color="inherit",
                white_space="pre-wrap",
            ),
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
        rx.vstack(
            rx.hstack(
                rx.spinner(size="2"),
                rx.text(
                    "Bikepacking assistant",
                    font_size="12px",
                    font_weight="700",
                    letter_spacing="0.12em",
                    text_transform="uppercase",
                    color="#64748b",
                ),
                spacing="3",
                align="center",
            ),
            rx.cond(
                BikepackingState.loading_stage_label != "",
                rx.text(
                    BikepackingState.loading_stage_label,
                    font_size="15px",
                    line_height="1.7",
                    color="#0f172a",
                    font_weight="700",
                ),
                rx.text(
                    "Working through the recommendation pipeline",
                    font_size="15px",
                    line_height="1.7",
                    color="#0f172a",
                    font_weight="700",
                ),
            ),
            rx.cond(
                BikepackingState.loading_stage_history.length() > 1,
                rx.hstack(
                    rx.foreach(
                        BikepackingState.loading_stage_history[-3:],
                        lambda item: pill(item["stage_label"]),
                    ),
                    spacing="2",
                    wrap="wrap",
                    width="100%",
                ),
                rx.fragment(),
            ),
            rx.text(
                "Working through the recommendation pipeline",
                font_size="13px",
                line_height="1.6",
                color="#64748b",
            ),
            spacing="2",
            align="start",
            width="100%",
        ),
        max_width="78%",
    )

def _composer() -> rx.Component:
    """Render the message composer."""
    return surface_card(
        rx.vstack(
            rx.text(
                "Ask your next question",
                font_size="13px",
                font_weight="700",
                letter_spacing="0.12em",
                text_transform="uppercase",
                color="#64748b",
            ),
            rx.text_area(
                value=BikepackingState.query,
                on_change=BikepackingState.set_query,
                placeholder="Ask about a full setup or a specific component…",
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
                    "Ask follow-up questions in the same thread.",
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
            width="100%",
            flex="1",
            min_height="0",
            overflow_y="auto",
            overflow_x="hidden",
            padding_right="6px",
        ),
        padding="22px",
        width="100%",
        flex="1",
        min_height="0",
        display="flex",
        flex_direction="column",
        overflow="hidden",
    )


@rx.page(route="/", title="Bikepacking Recommender")
def home() -> rx.Component:
    """Render the simplified chat-first bikepacking assistant."""
    return rx.box(
        rx.box(
            position="absolute",
            inset="0",
            background=PAGE_BG,
            z_index="-1",
        ),
        rx.box(
            rx.vstack(
                rx.box(_header(), flex_shrink="0"),
                _chat_thread(),
                rx.box(_composer(), flex_shrink="0"),
                rx.box(
                    rx.text(
                        "Grounded recommendations powered by the bikepacking backend.",
                        font_size="12px",
                        color="#64748b",
                        text_align="center",
                    ),
                    flex_shrink="0",
                ),
                spacing="4",
                align="stretch",
                width="100%",
                flex="1",
                min_height="0",
                overflow="hidden",
            ),
            width="100%",
            max_width="960px",
            margin_x="auto",
            padding_x=["14px", "18px", "24px"],
            padding_y=["16px", "22px", "28px"],
            height="100%",
            min_height="0",
            display="flex",
            flex_direction="column",
        ),
        height="100vh",
        min_height="100vh",
        position="relative",
        overflow="hidden",
    )