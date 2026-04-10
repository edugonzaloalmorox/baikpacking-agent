"""Reusable UI components for the bikepacking chat UI."""

from typing import Any, Iterable

import reflex as rx


PAGE_BG = "linear-gradient(180deg, #f8fafc 0%, #f7f8fb 46%, #eef2f7 100%)"

SURFACE_STYLE = {
    "background_color": "rgba(255, 255, 255, 0.76)",
    "border": "1px solid rgba(148, 163, 184, 0.16)",
    "box_shadow": "0 18px 60px rgba(15, 23, 42, 0.08)",
    "backdrop_filter": "blur(18px)",
    "border_radius": "28px",
}

USER_BUBBLE_STYLE = {
    "background_color": "#0f172a",
    "border": "1px solid #0f172a",
    "color": "#f8fafc",
}

ASSISTANT_BUBBLE_STYLE = {
    "background_color": "rgba(255, 255, 255, 0.84)",
    "border": "1px solid rgba(148, 163, 184, 0.18)",
    "color": "#0f172a",
    "box_shadow": "0 12px 40px rgba(15, 23, 42, 0.06)",
}

ERROR_BUBBLE_STYLE = {
    "background_color": "#fff7f7",
    "border": "1px solid rgba(239, 68, 68, 0.25)",
    "color": "#991b1b",
}

LABEL_STYLE = {
    "color": "#64748b",
    "font_size": "11px",
    "font_weight": "700",
    "letter_spacing": "0.14em",
    "text_transform": "uppercase",
}

TEXT_STYLE = {
    "color": "#0f172a",
    "font_size": "15px",
    "line_height": "1.7",
}

MUTED_STYLE = {
    "color": "#64748b",
    "font_size": "13px",
    "line_height": "1.65",
}


def _safe_display(value: Any, fallback: str = "—") -> rx.Component:
    """Render a value safely for Reflex without Python boolean checks.

    This helper avoids crashes caused by using Reflex Vars in Python truthiness
    expressions such as `if value`, `value and ...`, or `value.strip()`.
    """
    return rx.cond((value != None) & (value != ""), value, fallback)


def surface_card(*children: rx.Component, padding: str = "24px", **style: Any) -> rx.Component:
    """Render a soft product-style surface."""
    merged_style = {
        **SURFACE_STYLE,
        "width": "100%",
        "padding": padding,
        **style,
    }
    return rx.card(*children, size="4", **merged_style)


def bubble(role: str, *children: rx.Component, max_width: str = "78%", **style: Any) -> rx.Component:
    """Render a chat bubble for a given role."""
    bubble_style = (
        USER_BUBBLE_STYLE
        if role == "user"
        else ERROR_BUBBLE_STYLE
        if role == "error"
        else ASSISTANT_BUBBLE_STYLE
    )
    align = "flex-end" if role == "user" else "flex-start"

    merged_style = {
        **bubble_style,
        "padding": "18px 18px 16px 18px",
        "border_radius": "24px",
        "width": "100%",
        "max_width": max_width,
        **style,
    }

    return rx.box(
        rx.box(*children, **merged_style),
        width="100%",
        display="flex",
        justify_content=align,
    )


def pill(text: Any, accent: bool = False) -> rx.Component:
    """Render a compact capsule label."""
    background = "#0f172a" if accent else "#e2e8f0"
    color = "#f8fafc" if accent else "#0f172a"
    return rx.box(
        rx.text(_safe_display(text, ""), font_size="12px", font_weight="700", color=color),
        padding_x="10px",
        padding_y="5px",
        border_radius="999px",
        background_color=background,
        display="inline-flex",
        align_items="center",
    )


def section_heading(eyebrow: str, title: str, subtitle: str | None = None) -> rx.Component:
    """Render a clean section heading."""
    nodes: list[rx.Component] = [
        rx.text(eyebrow, **LABEL_STYLE),
        rx.heading(title, size="5", letter_spacing="-0.03em", color="#0f172a"),
    ]
    if subtitle:
        nodes.append(rx.text(subtitle, **MUTED_STYLE))
    return rx.vstack(*nodes, spacing="2", align="start", width="100%")


def prompt_chip(text: str, on_click: Any) -> rx.Component:
    """Render a clickable prompt suggestion."""
    return rx.button(
        text,
        on_click=on_click,
        variant="soft",
        color_scheme="gray",
        border_radius="999px",
        padding_x="14px",
        padding_y="10px",
        font_weight="600",
        white_space="normal",
        text_align="left",
    )


def detail_row(label: str, value: Any) -> rx.Component:
    """Render a compact detail row safely."""
    return rx.hstack(
        rx.text(label, **LABEL_STYLE),
        rx.text(
            _safe_display(value),
            font_size="14px",
            color="#0f172a",
            font_weight="600",
            text_align="right",
        ),
        justify="between",
        align="start",
        width="100%",
    )


def metric_tile(label: str, value: Any) -> rx.Component:
    """Render a compact metric tile safely."""
    return rx.box(
        rx.text(label, font_size="12px", font_weight="500", color="#64748b"),
        rx.text(
            _safe_display(value),
            font_size="18px",
            font_weight="700",
            color="#0f172a",
        ),
        padding="14px",
        border_radius="16px",
        background_color="#f8fafc",
        border="1px solid rgba(148, 163, 184, 0.12)",
    )


def key_value_grid(items: Iterable[tuple[str, Any]]) -> rx.Component:
    """Render a compact detail grid."""
    rows = [detail_row(label, value) for label, value in items]
    return rx.vstack(*rows, spacing="3", align="start", width="100%")


def message_text(value: Any, **style: Any) -> rx.Component:
    """Render message text safely."""
    merged_style = {**TEXT_STYLE, **style}
    return rx.text(_safe_display(value, ""), **merged_style)


def muted_text(value: Any, **style: Any) -> rx.Component:
    """Render muted text safely."""
    merged_style = {**MUTED_STYLE, **style}
    return rx.text(_safe_display(value, ""), **merged_style)