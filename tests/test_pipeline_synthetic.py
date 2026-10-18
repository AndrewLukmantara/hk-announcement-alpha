import shutil
from pathlib import Path

import yaml

from hkalpha.config import load_config
from hkalpha.pipeline import run

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_pipeline_recovers_planted_effects(tmp_path):
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["synthetic"] = {"n_stocks": 60, "seed": 3}
    (tmp_path / "data").mkdir()
    shutil.copy(ROOT / "data" / "universe.csv", tmp_path / "data" / "universe.csv")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    res = run(load_config(tmp_path / "config.yaml"), synthetic=True, regenerate=True)

    b = res["by_bucket"].set_index("bucket")
    assert b.loc["positive", "car_0_bps"] > 50 > -50 > b.loc["negative", "car_0_bps"]
    assert b.loc["positive", "drift_5_bps"] > 0 > b.loc["negative", "drift_5_bps"]
    reg = res["regression"]
    assert reg.params["kw_both"] > 0 and reg.params["kw_both:flow_z"] > 0
    assert (res["out_dir"] / "summary.md").exists()
