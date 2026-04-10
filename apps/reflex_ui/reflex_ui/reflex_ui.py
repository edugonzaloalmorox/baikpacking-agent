"""Main entrypoint for the bikepacking Reflex UI."""

import reflex as rx

from .pages.home import home  # noqa: F401  # Imported for page registration.


app = rx.App(
    theme=rx.theme(
        appearance="light",
        accent_color="teal",
        gray_color="slate",
        radius="large",
    ),
)
