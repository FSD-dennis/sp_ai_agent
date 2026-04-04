"""Render daily research reports from Jinja2 templates."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Environment, FileSystemLoader

from src.utils.logger import get_logger

log = get_logger(__name__)

_TEMPLATE_DIR = Path(__file__).parent
_TEMPLATE_NAME = "template.md.j2"


def generate_report(context: dict[str, Any]) -> str:
    """Render the markdown report from *context* dict."""
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        keep_trailing_newline=True,
    )
    template = env.get_template(_TEMPLATE_NAME)

    # Convert numpy arrays in transition matrix to readable string
    if "hmm_transition_matrix" in context and isinstance(
        context["hmm_transition_matrix"], (np.ndarray, list)
    ):
        mat = np.array(context["hmm_transition_matrix"])
        lines = []
        for row in mat:
            lines.append("  ".join(f"{v:.3f}" for v in row))
        context["hmm_transition_matrix"] = "\n".join(lines)

    return template.render(**context)


def save_report(content: str, output_dir: str | Path) -> str:
    """Write report markdown to ``output_dir/YYYY-MM-DD.md``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    path = output_dir / f"{stamp}.md"
    path.write_text(content, encoding="utf-8")
    log.info("Report saved → %s", path)
    return str(path)
