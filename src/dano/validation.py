"""Purged, predetermined year/grade validation; no sample-level random split."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Fold:
    name: str
    train: np.ndarray
    valid: np.ndarray
    purged: np.ndarray


def diagonal_folds(frame):
    """Each test fold has grade 9/10/11 in three DISTINCT years.

    Five cyclic folds cover every year/grade once when five seasons exist.
    All grades of all three test years are removed from training. This is
    deliberately stricter than merely splitting on a year--grade identifier.
    """
    years = sorted(frame.year.unique())
    if len(years) < 4:
        raise ValueError("At least four labeled years required for purged diagonal validation")
    result = []
    for i in range(len(years)):
        cases = [(years[(i + j) % len(years)], 9 + j) for j in range(3)]
        valid = np.array([idx for idx, row in frame.iterrows() if (row.year, row.grade) in cases])
        test_years = {int(frame.loc[idx, "year"]) for idx in valid}
        train = np.array([idx for idx, row in frame.iterrows() if row.year not in test_years])
        purged = np.array([idx for idx, row in frame.iterrows() if row.year in test_years and idx not in valid])
        fold = Fold(f"diagonal_{i}", train, valid, purged)
        assert_isolated(frame, fold, diagonal=True)
        result.append(fold)
    return result


def single_case_year_folds(frame):
    """Supplementary LOYO: each validation fold contains exactly one case.

    All three grades of the case's year are purged. Predictions from the three
    folds of one year are correlated; uncertainty is clustered by year.
    """
    result = []
    for idx, row in frame.iterrows():
        train = frame.index[frame.year != row.year].to_numpy()
        purged = frame.index[(frame.year == row.year) & (frame.index != idx)].to_numpy()
        fold = Fold(f"single_{int(row.year)}_{int(row.grade)}", train, np.array([idx]), purged)
        assert_isolated(frame, fold)
        result.append(fold)
    return result


def assert_isolated(frame, fold, diagonal=False):
    assert not set(fold.train) & set(fold.valid)
    assert not set(frame.loc[fold.train, "year"]) & set(frame.loc[fold.valid, "year"])
    if diagonal:
        assert len(frame.loc[fold.valid, "year"].unique()) == len(fold.valid) == 3
        assert set(frame.loc[fold.valid, "grade"]) == {9, 10, 11}
    assert len(fold.train) > 0


def expanding_folds(frame):
    """Forecast-direction sensitivity: only strictly earlier seasons train."""
    years = sorted(frame.year.unique())
    folds = []
    for year in years[2:]:
        for grade in (9, 10, 11):
            valid = frame.index[(frame.year == year) & (frame.grade == grade)].to_numpy()
            train = frame.index[frame.year < year].to_numpy()
            purged = frame.index[(frame.year >= year) & ~frame.index.isin(valid)].to_numpy()
            fold = Fold(f"forward_{year}_{grade}", train, valid, purged)
            assert_isolated(frame, fold)
            assert frame.loc[train, "year"].max() < year
            folds.append(fold)
    return folds
