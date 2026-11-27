import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from brent import paths, plots


def test_save_fig_writes_png_at_150_dpi(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FIGURES", tmp_path)
    plots.setup_style()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1])
    out = plots.save_fig(fig, "toy")
    assert out == tmp_path / "toy.png" and out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    from PIL import Image
    assert [round(v) for v in Image.open(out).info["dpi"]] == [150, 150]   # saved at 150 dpi (margins trimmed)
