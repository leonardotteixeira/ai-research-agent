"""Resolves institutional logo files for the cover page.

Provenance: the two files this project ships under `assets/logos/` were
supplied directly by the project's author (not downloaded from a
third-party site by this code), who identified them as the official
UNICAMP and FEAGRI marks. This module does not fetch, generate, or
otherwise fabricate a logo -- it only looks for files at known local
paths and returns what it finds. Missing files are not an error: the
cover page renders without them (see AcademicPDFRenderer), since a
missing institutional asset should degrade gracefully, not break
document generation.
"""

from pathlib import Path

DEFAULT_LOGOS_DIR = Path("assets/logos")

# Order matters: this is the order logos appear on the cover, top to
# bottom. Add a filename here (and drop the matching file into
# assets/logos/) to add another institutional mark -- nothing else needs
# to change.
_KNOWN_LOGO_FILENAMES = ("unicamp.png", "feagri.jpg")


def resolve_logo_paths(logos_dir: Path = DEFAULT_LOGOS_DIR) -> list[Path]:
    """Returns the known logo files that actually exist on disk, in
    display order. Never raises for a missing file or directory."""
    if not logos_dir.is_dir():
        return []
    return [path for name in _KNOWN_LOGO_FILENAMES if (path := logos_dir / name).is_file()]
