import pytest
from brent import paths


def test_repo_within_budget():
    assert paths.check_budget(raise_=False).empty


def test_oversized_file_flagged(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "big.csv").write_bytes(b"0" * (11 * 2**20))
    with pytest.raises(ValueError, match="big.csv"):
        paths.check_budget(root=tmp_path)


def test_dot_folders_ignored(tmp_path):
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "huge.bin").write_bytes(b"0" * (11 * 2**20))
    assert paths.check_budget(raise_=False, root=tmp_path).empty
