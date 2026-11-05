import numpy as np
import pandas as pd
import pytest
from dano.metrics import regression_metrics
from dano.validation import diagonal_folds, single_case_year_folds, expanding_folds


@pytest.fixture
def cases():
    return pd.DataFrame([dict(year=y, grade=g) for y in range(2021, 2026) for g in (9, 10, 11)])


def test_custom_metric_examples():
    assert regression_metrics([56], [57])["power_1_5"] == 1
    assert regression_metrics([56], [58])["power_1_5"] == pytest.approx(2 ** 1.5)
    with pytest.raises(ValueError):
        regression_metrics([56], [np.nan])


def test_diagonal_has_no_shared_year_and_covers_every_case(cases):
    folds = diagonal_folds(cases)
    assert len(folds) == 5
    assert sorted(np.concatenate([f.valid for f in folds])) == list(cases.index)
    for fold in folds:
        assert len(fold.train) == 6
        assert len(fold.purged) == 6
        assert len(cases.loc[fold.valid].year.unique()) == 3
        assert len(cases.loc[fold.valid].grade.unique()) == 3
        assert set(cases.loc[fold.train].year).isdisjoint(cases.loc[fold.valid].year)


def test_supplementary_and_forward_purge_all_grades(cases):
    for fold in single_case_year_folds(cases):
        assert len(fold.valid) == 1
        assert len(fold.train) == 12
        assert set(cases.loc[fold.train].year).isdisjoint(cases.loc[fold.valid].year)
    for fold in expanding_folds(cases):
        assert cases.loc[fold.train].year.max() < cases.loc[fold.valid].year.min()
