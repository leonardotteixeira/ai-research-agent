from pathlib import Path

from app.academic.assets import resolve_logo_paths


class TestResolveLogoPaths:
    def test_missing_directory_returns_empty_list(self, tmp_path: Path):
        assert resolve_logo_paths(tmp_path / "does-not-exist") == []

    def test_empty_directory_returns_empty_list(self, tmp_path: Path):
        assert resolve_logo_paths(tmp_path) == []

    def test_returns_known_files_in_display_order(self, tmp_path: Path):
        (tmp_path / "feagri.jpg").write_bytes(b"fake-jpg")
        (tmp_path / "unicamp.png").write_bytes(b"fake-png")

        paths = resolve_logo_paths(tmp_path)

        assert [p.name for p in paths] == ["unicamp.png", "feagri.jpg"]

    def test_gracefully_skips_a_missing_individual_file(self, tmp_path: Path):
        (tmp_path / "unicamp.png").write_bytes(b"fake-png")

        paths = resolve_logo_paths(tmp_path)

        assert [p.name for p in paths] == ["unicamp.png"]

    def test_unknown_files_in_the_directory_are_ignored(self, tmp_path: Path):
        (tmp_path / "unicamp.png").write_bytes(b"fake-png")
        (tmp_path / "random_logo.png").write_bytes(b"fake-png")

        paths = resolve_logo_paths(tmp_path)

        assert [p.name for p in paths] == ["unicamp.png"]

    def test_real_project_assets_directory_has_both_logos(self):
        """The two files this project actually ships under
        assets/logos/ -- provided directly by the project's author, who
        identified them as the official UNICAMP/FEAGRI marks (see
        app/academic/assets.py's module docstring for provenance)."""
        paths = resolve_logo_paths()
        names = {p.name for p in paths}
        assert names == {"unicamp.png", "feagri.jpg"}
