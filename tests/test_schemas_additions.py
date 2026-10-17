"""Master additions to the data dictionary after Task 1: contributions.csv and spotcheck/items.json."""

from pathlib import Path

import pandas as pd

from core import schemas


def test_contributions_table_is_registered_and_validates(tmp_path):
    assert schemas.CONTRIBUTIONS in schemas.TABLES
    assert schemas.contributions_csv(tmp_path) == tmp_path / "contributions.csv"
    df = pd.DataFrame([{"case_id": "BACH_0001", "split": "test", "model": "cb_text", "feature": "invasion__top3",
                        "concept": "invasion", "source": "text", "value": 1.9, "value_std": 1.2, "contribution": 0.44}])
    assert schemas.frame_problems(schemas.CONTRIBUTIONS, df) == []
    bad = df.assign(source="guess")
    assert schemas.frame_problems(schemas.CONTRIBUTIONS, bad)


def test_spotcheck_items_spec_is_registered(tmp_path):
    assert schemas.SPOTCHECK_ITEMS in schemas.JSON_SPECS
    assert schemas.spotcheck_items_json(tmp_path) == Path(tmp_path) / "spotcheck" / "items.json"
    assert schemas.SPOTCHECK_ITEMS.problems({"created_utc": "x", "synthetic": True, "items": []}) == []
    assert schemas.SPOTCHECK_ITEMS.problems({"items": []})
