"""Shared Plotly style defaults for consistent benchmark figures."""

from __future__ import annotations

import plotly.io as pio

PASTEL_COLORWAY = [
    "#8ecae6",
    "#ffb4a2",
    "#bde0fe",
    "#cdb4db",
    "#b9fbc0",
    "#ffd6a5",
    "#f1c0e8",
    "#a0c4ff",
]

DEFAULT_FONT_FAMILY = "Inter, Arial, Helvetica, sans-serif"


def register_flax_kmeans_template() -> None:
    """Register a project-specific Plotly template if missing."""
    if "flax_kmeans_pastel" in pio.templates:
        return

    pio.templates["flax_kmeans_pastel"] = {
        "layout": {
            "colorway": PASTEL_COLORWAY,
            "font": {"family": DEFAULT_FONT_FAMILY, "size": 13},
            "title": {"font": {"size": 18}},
            "paper_bgcolor": "#ffffff",
            "plot_bgcolor": "#ffffff",
            "xaxis": {
                "showgrid": True,
                "gridcolor": "#e9ecef",
                "gridwidth": 1,
                "zeroline": False,
            },
            "yaxis": {
                "showgrid": True,
                "gridcolor": "#e9ecef",
                "gridwidth": 1,
                "zeroline": False,
            },
            "legend": {"title": {"font": {"size": 12}}},
        }
    }


def use_default_template() -> None:
    """Activate the project template globally for this Python process."""
    register_flax_kmeans_template()
    pio.templates.default = "flax_kmeans_pastel"
