"""Reusable UI components for the bikepacking Reflex app."""



from typing import Any, Iterable

import reflex as rx


CARD_STYLE = {
    "background_color": "rgba(255, 255, 255, 0.82)",
    "border": "1px solid rgba(148, 163, 184, 0.18)",
    "box_shadow": "0 18px 60px rgba(15, 23, 42, 0.08)",
    "backdrop_filter": "blur(18px)",
}

LABEL_STYLE = {
    "color": "#64748b",
    "font_size": "12px",
    "font_weight": "700",
    "letter_spacing": "0.12em",
    "text_transform": "uppercase",
}

VALUE_STYLE = {
    "color": "#0f172a",
    "font_size": "16px",
    "font_weight": "600",
    "line_height": "1.4",
}

BODY_STYLE = {
    "color": "#475569",
    "font_size": "14px",
    "line_height": "1.7",
}

def card(*children: rx.Component, padding: str = "24px", **style: Any) -> rx.Component:
    """Create a consistent glassy card surface."""
    merged_style = {
        **CARD_STYLE,
        "width": "100%",
        "padding": padding,
        "border_radius": "24px",
        **style,
    }
    return rx.card(
        *children,
        size="4",
        **merged_style,
    )


def section_heading(eyebrow: str, title: str, subtitle: str | None = None) -> rx.Component:
    """Render a compact section heading."""
    items: list[rx.Component] = [
        rx.text(eyebrow, **LABEL_STYLE),
        rx.heading(title, size="5", letter_spacing="-0.03em", color="#0f172a"),
    ]
    if subtitle:
        items.append(rx.text(subtitle, **BODY_STYLE))
    return rx.vstack(*items, spacing="2", align="start", width="100%")


def stat_card(label: str, value: str, helper: str | None = None) -> rx.Component:
    """Render a compact metric card."""
    children: list[rx.Component] = [
        rx.text(label, **LABEL_STYLE),
        rx.text(value, **VALUE_STYLE),
    ]
    if helper:
        children.append(rx.text(helper, color="#64748b", font_size="13px"))
    return card(rx.vstack(*children, spacing="2", align="start"), padding="18px")


def field_card(label: str, value) -> rx.Component:
    """Render a key/value tile for setup fields."""
    return rx.box(
        rx.text(label, **LABEL_STYLE),
        rx.text(
            rx.cond((value != None) & (value != ""), value, "—"),
            **VALUE_STYLE,
        ),
        padding="16px",
        border_radius="18px",
        background_color="#f8fafc",
        border="1px solid rgba(148, 163, 184, 0.12)",
    )


def pill(text: str, accent: bool = False) -> rx.Component:
    """Render a subtle capsule label."""
    background = "#0f172a" if accent else "#e2e8f0"
    color = "#f8fafc" if accent else "#0f172a"
    return rx.box(
        rx.text(text, font_size="12px", font_weight="700", color=color),
        padding_x="10px",
        padding_y="5px",
        border_radius="999px",
        background_color=background,
        display="inline-flex",
        align_items="center",
    )


def chip_row(items: Iterable[str]) -> rx.Component:
    """Render a horizontal row of small example chips."""
    chips = [
        pill(item)
        for item in items
        if isinstance(item, str) and item.strip()
    ]
    return rx.hstack(*chips, spacing="2", wrap="wrap")


def empty_state(message: str, hint: str | None = None) -> rx.Component:
    """Render a friendly empty state."""
    children: list[rx.Component] = [
        rx.text(message, **VALUE_STYLE),
    ]
    if hint:
        children.append(rx.text(hint, **BODY_STYLE))
    return card(rx.vstack(*children, spacing="2", align="start"), padding="28px")
