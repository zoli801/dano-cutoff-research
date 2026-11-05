import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from dano.dataset import ITEM_ALIGNMENT, split_statements
from dano.features import Corpus, Representation

ROOT = Path(__file__).resolve().parents[1]


def test_physical_problem_alignment_covers_each_score_column_once():
    for year, expected in [(2021, 25), (2022, 18), (2023, 26), (2024, 20), (2025, 20)]:
        flattened = [item for group in ITEM_ALIGNMENT[year] for item in group]
        assert sorted(flattened) == list(range(1, expected+1))
        assert len(set(flattened)) == len(flattened)


def test_answer_annotations_do_not_enter_statement_text():
    sample = '\n'.join(f'Вопрос {i} (Задание оценивается в 2 балла)\nУсловие задачи номер {i}.\nОтвет: секрет\nОбъяснение ответа.' for i in range(1, 26))
    records = split_statements(sample, expected=25, legacy=True)
    assert len(records) == 25
    assert all('секрет' not in r['text'] and 'Объяснение' not in r['text'] for r in records)
    bad_sample = sample.replace('Вопрос 17 ', 'Пропущен 17 ')
    with pytest.raises(ValueError):
        split_statements(bad_sample, expected=25, legacy=True)


@pytest.mark.skipif(not (ROOT/'data/interim/rubert_embeddings.npy').exists(), reason='requires collected corpus')
def test_supervision_weights_and_fold_vocabulary():
    corpus = Corpus()
    train = corpus.supervised({2021, 2022})
    assert set(train.year) == {2021, 2022}
    assert np.allclose(train.groupby(['year', 'grade', 'item']).sample_weight.sum(), 1)
    banned = {corpus.problems.iloc[0].family_hash}
    assert not set(corpus.supervised({2021, 2022}, banned).family_hash) & banned
    # Sentinel appears ONLY in a future/held-out document: fitting IDF/SVD must
    # not add it to a training fold vocabulary.
    future = corpus.problems.index[corpus.problems.year == 2026][0]
    corpus.problems.loc[future, 'text'] += ' validationonlysentinelxyz'
    rep = Representation(corpus, 'tfidf', corpus.years_indices({2022}))
    assert 'validationonlysentinelxyz' not in rep.word.vocabulary_


@pytest.mark.skipif(not (ROOT/'results/validation_predictions.csv').exists(), reason='requires benchmark')
def test_all_architectures_use_identical_cases_and_no_legacy_target():
    frame = pd.read_csv(ROOT/'results/validation_predictions.csv')
    for protocol, data in frame.groupby('protocol'):
        sets = [set(zip(group.year, group.grade)) for _, group in data.groupby('model')]
        assert all(s == sets[0] for s in sets)
        assert not data.duplicated(['model','year','grade']).any()
        assert np.isfinite(data.prediction).all()
        assert data.prediction.between(0,100).all()
        assert data.year.min() >= 2022
    for fold in json.loads((ROOT/'results/splits.json').read_text()):
        train_years = {r['year'] for r in fold['train']}
        valid_years = {r['year'] for r in fold['validation']}
        assert train_years.isdisjoint(valid_years)
        assert len(valid_years) == len(fold['validation'])
        assert len({r['grade'] for r in fold['validation']}) == len(fold['validation'])
