import tempfile
from pathlib import Path

from omniopd.provenance import fingerprint_code_tree, fingerprint_path


def test_directory_fingerprint_is_stable_and_content_sensitive():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "a.txt").write_text("a", encoding="utf-8")
        (root / "b.txt").write_text("b", encoding="utf-8")
        first = fingerprint_path(root)
        second = fingerprint_path(root)
        assert first["tree_sha256"] == second["tree_sha256"]
        (root / "b.txt").write_text("changed", encoding="utf-8")
        assert fingerprint_path(root)["tree_sha256"] != first["tree_sha256"]


def test_code_fingerprint_excludes_legacy_and_cache_files():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "src").mkdir()
        (root / "src" / "code.py").write_text("x = 1\n", encoding="utf-8")
        (root / "legacy").mkdir()
        (root / "legacy" / "old.py").write_text("broken", encoding="utf-8")
        before = fingerprint_code_tree(root)
        (root / "legacy" / "old.py").write_text("changed", encoding="utf-8")
        assert fingerprint_code_tree(root) == before
        (root / "src" / "code.py").write_text("x = 2\n", encoding="utf-8")
        assert fingerprint_code_tree(root)["tree_sha256"] != before["tree_sha256"]
