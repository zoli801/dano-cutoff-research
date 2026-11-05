"""Train-fold-only lexical features and deterministic problem/set pooling."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

ROOT = Path(__file__).resolve().parents[2]
TOPICS = {
    "probability": r"вероятн|комбинац|случайн|независим",
    "statistics": r"дисперс|медиан|средн|стандартн|квартил",
    "hypothesis": r"гипотез|p.value|значим|тест|доверительн",
    "regression": r"регресси|коэффициент|корреляц",
    "ml": r"модел|классификац|accuracy|precision|recall|алгоритм|ошибк",
    "visual": r"график|диаграмм|гистограмм|рисунк|ящик",
    "data_file": r"\.csv|\.xlsx?|файл|баз[аеуы] данных",
    "causality": r"причин|влиян|смещени|selection|bias|эксперимент",
    "economics": r"доход|зарплат|ввп|налог|выручк|кредит|цен[ауы]",
}


def text_features(text):
    low = text.lower()
    words = re.findall(r"\w+", low)
    length = max(len(text), 1)
    values = {
        "log_characters": np.log1p(length), "log_words": np.log1p(len(words)),
        "digit_fraction": sum(c.isdigit() for c in text) / length,
        "math_symbol_fraction": sum(c in "=<>±σμ∑∫√^" for c in text) / length,
        "question_count": text.count("?"),
        "mean_word_length": np.mean([len(w) for w in words]) if words else 0,
        "latin_fraction": len(re.findall(r"[a-z]", low)) / length,
    }
    values.update({"topic_" + name: len(re.findall(pattern, low)) / max(len(words), 1) * 100
                   for name, pattern in TOPICS.items()})
    return values


class Corpus:
    def __init__(self):
        path = ROOT / "data/processed/problem_texts.jsonl"
        self.problems = pd.DataFrame([json.loads(s) for s in path.read_text().splitlines()])
        self.numeric = pd.DataFrame([text_features(t) for t in self.problems.text])
        self.embeddings = np.load(ROOT / "data/interim/rubert_embeddings.npy")
        order = json.loads((ROOT / "data/interim/embedding_order.json").read_text())
        assert [r["problem_id"] for r in order] == self.problems.problem_id.tolist()
        assert [r["text_sha256"] for r in order] == [hashlib.sha256(t.encode()).hexdigest() for t in self.problems.text]
        assert self.embeddings.shape[0] == len(self.problems)
        self.labels = pd.read_csv(ROOT / "data/processed/difficulty_labels.csv")

    def years_indices(self, years):
        return self.problems.index[self.problems.year.isin(years)].to_numpy()

    def supervised(self, years, banned_hashes=()):
        frame = self.problems[self.problems.year.isin(years)].reset_index(names="problem_index")
        frame = frame[~frame.family_hash.isin(banned_hashes)]
        frame = frame.merge(self.labels, on=["year", "item"], validate="many_to_many")
        frame["sample_weight"] = 1 / frame.groupby(["year", "grade", "item"])["variant"].transform("nunique")
        return frame.reset_index(drop=True)


class Representation:
    def __init__(self, corpus, kind, train_problem_indices):
        self.kind, self.corpus = kind, corpus
        self.numeric_names = corpus.numeric.columns.tolist()
        if kind == "tfidf":
            texts = corpus.problems.loc[train_problem_indices, "text"].tolist()
            self.word = TfidfVectorizer(ngram_range=(1, 2), max_features=3000,
                                        sublinear_tf=True, min_df=1, dtype=np.float32)
            self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                        max_features=3000, sublinear_tf=True, dtype=np.float32)
            train = hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()
            components = min(24, train.shape[0]-1, train.shape[1]-1)
            self.svd = TruncatedSVD(n_components=components, random_state=20261105)
            self.svd.fit(train)
            all_texts = corpus.problems.text.tolist()
            self.semantic = self.svd.transform(hstack([self.word.transform(all_texts), self.char.transform(all_texts)]).tocsr())
        elif kind == "rubert":
            self.semantic = corpus.embeddings.copy()
        else:
            raise ValueError(kind)
        self.problem_matrix = np.c_[self.semantic, corpus.numeric.to_numpy(float)]
        self.problem_names = [f"{kind}_{i:03d}" for i in range(self.semantic.shape[1])] + self.numeric_names

    def problem_features(self, supervised):
        return np.c_[self.problem_matrix[supervised.problem_index.to_numpy(int)],
                     supervised.grade.to_numpy(float)-10]

    def set_features(self, cases):
        rows = []
        for row in cases.itertuples():
            idx = self.corpus.years_indices([row.year])
            semantic = self.semantic[idx]
            numeric = self.corpus.numeric.iloc[idx].to_numpy(float)
            rows.append(np.r_[semantic.mean(axis=0), numeric.mean(axis=0),
                              numeric.std(axis=0), self.corpus.problems.loc[idx, "item"].nunique(), row.grade-10])
        names = [f"mean_{self.kind}_{i:03d}" for i in range(self.semantic.shape[1])]
        names += ["mean_"+name for name in self.numeric_names]
        names += ["std_"+name for name in self.numeric_names] + ["problem_count", "grade_minus_10"]
        return np.asarray(rows, dtype=float), names
