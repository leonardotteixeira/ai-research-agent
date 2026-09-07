"""Thin CLI layer over the existing domain services.

This package only wires together (composition.py) and formats/parses
(main.py, formatting.py) — no Agent, tool, synthesis, persistence, or
Markdown logic lives here. See app/cli/main.py for the entry point:

    python -m app.cli --help
"""
