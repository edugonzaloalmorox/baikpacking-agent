"""Home page for the bikepacking recommendation UI."""



import reflex as rx

from apps.reflex_ui.reflex_ui.components import BODY_STYLE, card, chip_row, empty_state, field_card, pill, section_heading, stat_card
from apps.reflex_ui.reflex_ui.state import BikepackingState


SAMPLE_QUERIES = [
    "What tyres do you recommend for Atlas Mountain Race?",
    "What setup works best for Tour Divide?",
    "Need bikepacking bags for Transiberica",
]


def _shell_background() -> rx.Component:
    return rx.box(
        rx.box(
            position="absolute",
            top="0",
            left="0",
            width="40%",
            height="280px",
            background="radial-gradient(circle at top left, rgba(45, 212, 191, 0.16) 0%, rgba(45, 212, 191, 0.04) 30%, transparent 70%)",
            pointer_events="none",
        ),
        rx.box(
            position="absolute",
            right="0",
            top="120px",
            width="34%",
            height="320px",
            background="radial-gradient(circle at top right, rgba(148, 163, 184, 0.12) 0%, rgba(148, 163, 184, 0.02) 32%, transparent 75%)",
            pointer_events="none",
        ),
        position="absolute",
        inset="0",
        z_index="-1",
        background="linear-gradient(180deg, #f8fafc 0%, #f5f7fb 50%, #eef2f7 100%)",
    )


def _header() -> rx.Component:
    return rx.hstack(
        rx.vstack(
            rx.hstack(
                pill("Bikepacking Recommender", accent=True),
                pill("FastAPI"),
                spacing="2",
                wrap="wrap",
            ),
            rx.heading("Grounded setup recommendations for bikepacking events", size="7", letter_spacing="-0.04em"),
            rx.text(
                "Ask for event-specific tyres, bags, drivetrain, or full setups. The UI calls the backend API and renders the resolved event, evidence, policy, and debug trace.",
                max_width="760px",
                color="#475569",
                line_height="1.7",
            ),
            spacing="4",
            align="start",
            flex="1",
        ),
        card(
            rx.vstack(
                rx.text("API target", **BODY_STYLE),
                rx.text(BikepackingState.api_base_url, font_family="monospace", font_size="14px", color="#0f172a"),
                rx.text("Server-side request from the Reflex backend.", color="#64748b", font_size="13px"),
                spacing="2",
                align="start",
            ),
            padding="18px",
            width="auto",
            min_width="280px",
        ),
        spacing="4",
        align="start",
        justify="between",
        width="100%",
        wrap="wrap",
    )


def _query_panel() -> rx.Component:
    return card(
        rx.vstack(
            section_heading("Input", "Generate a recommendation", "Enter a question and the app will fetch a grounded answer from the backend."),
            rx.text_area(
                value=BikepackingState.query,
                on_change=BikepackingState.set_query,
                placeholder="What tyres do you recommend for Atlas Mountain Race?",
                width="100%",
                min_height="150px",
                padding="18px",
                border_radius="18px",
                background_color="#ffffff",
                border="1px solid rgba(148, 163, 184, 0.18)",
                color="#0f172a",
                font_size="16px",
            ),
            rx.hstack(
                rx.button(
                    "Generate recommendation",
                    on_click=BikepackingState.submit_query,
                    disabled=BikepackingState.loading,
                    background_color="#0f172a",
                    color="#f8fafc",
                    padding_x="22px",
                    padding_y="14px",
                    border_radius="14px",
                    font_weight="700",
                ),
                rx.text(
                    "The request includes debug traces so the result can be inspected without a second round-trip.",
                    color="#64748b",
                    font_size="13px",
                    line_height="1.6",
                ),
                spacing="4",
                align="center",
                justify="between",
                width="100%",
                wrap="wrap",
            ),
            chip_row(SAMPLE_QUERIES),
            spacing="4",
            align="start",
            width="100%",
        ),
        padding="24px",
    )


def _loading_banner() -> rx.Component:
    return rx.cond(
        BikepackingState.loading,
        card(
            rx.hstack(
                rx.spinner(size="3"),
                rx.vstack(
                    rx.text("Generating recommendation", font_weight="700", color="#0f172a"),
                    rx.text("Calling the bikepacking API and waiting for grounded output.", color="#64748b", font_size="13px"),
                    spacing="1",
                    align="start",
                ),
                spacing="3",
                align="center",
            ),
            padding="18px",
        ),
        rx.box(),
    )


def _error_banner() -> rx.Component:
    return rx.cond(
        BikepackingState.error != "",
        card(
            rx.vstack(
                rx.text("Could not generate a recommendation", font_weight="700", color="#991b1b"),
                rx.text(BikepackingState.error, color="#7f1d1d", line_height="1.6"),
                spacing="2",
                align="start",
            ),
            padding="20px",
            border="1px solid rgba(239, 68, 68, 0.22)",
            background_color="#fff7f7",
        ),
        rx.box(),
    )


def _result_summary() -> rx.Component:
    return card(
        rx.vstack(
            section_heading("Recommendation", "Summary and reasoning", "The summary is the primary answer; reasoning explains the grounding."),
            rx.box(
                rx.text(BikepackingState.recommendation_summary, font_size="20px", font_weight="700", color="#0f172a", line_height="1.5"),
                padding="18px",
                border_radius="18px",
                background_color="#f8fafc",
                border="1px solid rgba(148, 163, 184, 0.12)",
                width="100%",
            ),
            rx.text(BikepackingState.recommendation_reasoning, color="#475569", line_height="1.75", white_space="pre-wrap"),
            spacing="4",
            align="start",
            width="100%",
        ),
        padding="24px",
    )


def _compact_card(title: str, primary: rx.Component, secondary: rx.Component) -> rx.Component:
    return card(
        rx.vstack(
            rx.text(title, font_size="12px", font_weight="700", letter_spacing="0.12em", text_transform="uppercase", color="#64748b"),
            primary,
            secondary,
            spacing="2",
            align="start",
        ),
        padding="18px",
    )


def _resolved_event_panel() -> rx.Component:
    return _compact_card(
        "Resolved event",
        rx.text(BikepackingState.resolved_event_name, font_size="18px", font_weight="700", color="#0f172a"),
        rx.hstack(
            pill(BikepackingState.resolved_event_match_type),
            pill(BikepackingState.resolved_event_confidence),
            spacing="2",
            wrap="wrap",
        ),
    )


def _intent_panel() -> rx.Component:
    return _compact_card(
        "Intent",
        rx.text(BikepackingState.intent_component, font_size="18px", font_weight="700", color="#0f172a"),
        rx.hstack(
            pill(BikepackingState.intent_confidence),
            pill("component"),
            spacing="2",
            wrap="wrap",
        ),
    )


def _setup_panel() -> rx.Component:
    return card(
        rx.vstack(
            section_heading("Recommendation setup", "Recommended equipment", "The UI expands the compact setup into a clean, readable grid."),
            rx.grid(
                field_card("Bike", BikepackingState.setup_bike),
                field_card("Wheels", BikepackingState.setup_wheels),
                field_card("Tyres", BikepackingState.setup_tyres),
                field_card("Drivetrain", BikepackingState.setup_drivetrain),
                field_card("Bags", BikepackingState.setup_bags),
                field_card("Sleep system", BikepackingState.setup_sleep_system),
                field_card("Lighting", BikepackingState.setup_lighting),
                field_card("Navigation", BikepackingState.setup_navigation),
                field_card("Water capacity", BikepackingState.setup_water_capacity),
                field_card("Notes", BikepackingState.setup_notes),
                columns="repeat(auto-fit, minmax(200px, 1fr))",
                gap="12px",
                width="100%",
            ),
            spacing="4",
            align="start",
            width="100%",
        ),
        padding="24px",
    )


def _evidence_panel() -> rx.Component:
    return card(
        rx.vstack(
            section_heading("Evidence", "Signals and support", "These metrics help explain how strong the retrieval was."),
            rx.grid(
                stat_card("Rider count", BikepackingState.evidence_rider_count),
                stat_card("Component hits", BikepackingState.evidence_component_hit_count),
                stat_card("Strength", BikepackingState.evidence_strength),
                stat_card("Consistency", BikepackingState.evidence_consistency),
                columns="repeat(auto-fit, minmax(160px, 1fr))",
                gap="12px",
                width="100%",
            ),
            spacing="4",
            align="start",
            width="100%",
        ),
        padding="24px",
    )


def _policy_panel() -> rx.Component:
    return card(
        rx.vstack(
            section_heading("Policy", "Response mode", "The policy explains how strongly the writer may speak about the event and gear."),
            rx.hstack(
                pill(BikepackingState.policy_mode, accent=True),
                pill(BikepackingState.policy_allow_event_specific_claims),
                pill(BikepackingState.policy_allow_specific_specs),
                pill(BikepackingState.policy_allow_specific_brands),
                spacing="2",
                wrap="wrap",
            ),
            rx.text(BikepackingState.policy_notes, color="#475569", line_height="1.7", white_space="pre-wrap"),
            spacing="3",
            align="start",
            width="100%",
        ),
        padding="24px",
    )


def _debug_panel() -> rx.Component:
    return rx.cond(
        BikepackingState.has_debug,
        card(
            rx.vstack(
                rx.hstack(
                    section_heading("Debug", "Retrieval trace", "Collapsed by default to keep the UI clean."),
                    rx.button(
                        rx.cond(BikepackingState.show_debug, "Hide debug", "Show debug"),
                        on_click=BikepackingState.toggle_debug,
                        variant="soft",
                        color_scheme="gray",
                    ),
                    justify="between",
                    align="start",
                    width="100%",
                ),
                rx.cond(
                    BikepackingState.show_debug,
                    rx.vstack(
                        rx.box(
                            rx.text("Retrieval plan", font_weight="700", color="#0f172a"),
                            rx.text(BikepackingState.debug_retrieval_plan, font_family="monospace", font_size="12px", color="#334155", white_space="pre-wrap"),
                            padding="16px",
                            border_radius="16px",
                            background_color="#f8fafc",
                            border="1px solid rgba(148, 163, 184, 0.12)",
                            width="100%",
                        ),
                        rx.box(
                            rx.text("Trace", font_weight="700", color="#0f172a"),
                            rx.text(BikepackingState.debug_trace, font_family="monospace", font_size="12px", color="#334155", white_space="pre-wrap"),
                            padding="16px",
                            border_radius="16px",
                            background_color="#f8fafc",
                            border="1px solid rgba(148, 163, 184, 0.12)",
                            width="100%",
                        ),
                        spacing="3",
                        align="start",
                        width="100%",
                    ),
                    rx.box(),
                ),
                spacing="3",
                align="start",
                width="100%",
            ),
            padding="24px",
        ),
        rx.box(),
    )


def _results_section() -> rx.Component:
    return rx.cond(
        BikepackingState.has_response,
        rx.vstack(
            _result_summary(),
            rx.grid(
                _resolved_event_panel(),
                _intent_panel(),
                columns="repeat(auto-fit, minmax(280px, 1fr))",
                gap="12px",
                width="100%",
            ),
            _setup_panel(),
            rx.grid(
                _evidence_panel(),
                _policy_panel(),
                columns="repeat(auto-fit, minmax(320px, 1fr))",
                gap="12px",
                width="100%",
            ),
            _debug_panel(),
            spacing="4",
            align="start",
            width="100%",
        ),
        empty_state(
            "Your recommendation will appear here.",
            "The backend response is rendered as structured cards, including the resolved event, evidence, policy, and optional debug trace.",
        ),
    )


@rx.page(route="/", title="Bikepacking Recommender")
def home() -> rx.Component:
    """Render the main bikepacking recommendation screen."""
    return rx.box(
        _shell_background(),
        rx.container(
            rx.vstack(
                _header(),
                _query_panel(),
                _loading_banner(),
                _error_banner(),
                _results_section(),
                rx.text(
                    "Powered by the FastAPI recommendation backend and rendered locally in Reflex.",
                    color="#64748b",
                    font_size="12px",
                    text_align="center",
                    width="100%",
                ),
                spacing="5",
                align="stretch",
                width="100%",
            ),
            max_width="1200px",
            padding_x=["16px", "20px", "28px"],
            padding_y=["20px", "28px", "40px"],
        ),
        min_height="100vh",
        position="relative",
        overflow="hidden",
    )
