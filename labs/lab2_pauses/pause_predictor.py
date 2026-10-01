"""Pause predictor — ML-based implementation for lab 2 (v4: CatBoost + POS).

Использует:
  - data/pause_clf_v4.pkl  — CatBoostClassifier + подобранный порог
  - data/pause_reg_v4.pkl  — CatBoostRegressor длительности
  - features_v4.build_features_v4 — единый построитель признаков

Run as a script to score the predictor on the prepared data::

    python pause_predictor.py
"""
import csv
from pathlib import Path

import numpy as np
import tqdm
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error

from features_v4 import build_features_v4, CAT_FEATURES

HERE = Path(__file__).resolve().parent
PAUSE_PREDICTOR_DATA = HERE / 'data' / 'RUSLAN_pause_metadata.csv'
CLF_PATH = HERE / 'data' / 'pause_clf_v4.pkl'
REG_PATH = HERE / 'data' / 'pause_reg_v4.pkl'


class PausePredictor():
    """Predicts where pauses fall in a sentence and how long they are.

    Input — последовательность токенов `label_raw`:
        ["Я", "вышел", "из", "дома,", "когда", "стемнело."]
    """

    def __init__(self, clf_path=CLF_PATH, reg_path=REG_PATH):
        import pickle
        with open(clf_path, 'rb') as f:
            payload = pickle.load(f)
        self.clf = payload['model']
        self.threshold = float(payload['threshold'])
        with open(reg_path, 'rb') as f:
            self.reg = pickle.load(f)

    def _build_features(self, tokens) -> pd.DataFrame:
        """Оборачивает один список токенов в DataFrame и строит фичи v4."""
        tokens = [str(t) for t in tokens]
        n = len(tokens)
        # Синтетический df: один utterance с id='_'
        dummy = pd.DataFrame({
            'id': ['_'] * n,
            'label_raw': tokens,
        })
        return build_features_v4(dummy)

    def predict(self, tokens):
        """Decide for each token whether a pause follows it.

        Returns:
            is_pause (int array), pause_duration (float array).
        """
        tokens = list(tokens)
        n = len(tokens)
        if n == 0:
            return np.zeros(0, int), np.zeros(0, float)

        X = self._build_features(tokens)
        proba = self.clf.predict_proba(X)[:, 1]
        is_pause = (proba > self.threshold).astype(int)

        pause_duration = np.zeros(n, float)
        idx = np.where(is_pause == 1)[0]
        if len(idx) > 0:
            dur = self.reg.predict(X.iloc[idx])
            dur = np.clip(dur, 0.01, 2.0)
            pause_duration[idx] = dur

        return is_pause, pause_duration

    def predict_durations(self, tokens):
        """Вставляет `<SIL>` после каждой предсказанной паузы.

        Returns:
            tokens_w_pauses — токены с вставками `<SIL>`,
            durations — по одному значению на выходной токен: секунды для `<SIL>`,
            -1.0 для слов.
        """
        def expand_is_pause(token, flag):
            return [token, '<SIL>'] if bool(flag) else [token]

        def expand_dur(d):
            return [-1., d] if d > 0. else [-1.]

        is_pause, durations = self.predict(tokens)
        tokens_w_pauses = np.concatenate(
            [expand_is_pause(a, b) for a, b in zip(tokens, is_pause)]
        )
        durations_w_pauses = np.concatenate(
            [expand_dur(d) for d in durations]
        ).astype(np.float32)

        return tokens_w_pauses, durations_w_pauses


def calc_metrics(df: pd.DataFrame) -> None:
    rec = recall_score(df.is_pause_after, df.is_pause_hat)
    prc = precision_score(df.is_pause_after, df.is_pause_hat)
    f1 = f1_score(df.is_pause_after, df.is_pause_hat)

    tp = df[(df.is_pause_after == 1) & (df.is_pause_hat == 1)]
    if len(tp) == 0:
        print(f'PRC: {prc:.4f}, REC: {rec:.4f}, F1: {f1:.4f}; MAE: n/a')
    else:
        mae = mean_absolute_error(tp.pause_duration, tp.pause_duration_hat)
        print(f'PRC: {prc:.4f}, REC: {rec:.4f}, F1: {f1:.4f}; MAE: {mae:.4f};')


def test_pause_predictor() -> None:
    pause_df = pd.read_csv(PAUSE_PREDICTOR_DATA, sep='|', quoting=csv.QUOTE_NONE)
    pp = PausePredictor()

    lens = {i: l for i, l in pause_df.groupby('id').count()
            .reset_index(drop=False)[['id', 'label']].values}

    is_pause_after_hat = []
    pause_duration_hat = []
    idx = 0
    for i, is_last in tqdm.tqdm(pause_df[['id', 'is_last_word']].values):
        if not is_last:
            continue
        sentence = pause_df.iloc[idx:idx + lens[i]]
        idx += lens[i]
        is_pause_hat, pause_dur_hat = pp.predict(sentence.label_raw.values)
        is_pause_after_hat += list(is_pause_hat)
        pause_duration_hat += list(pause_dur_hat)

    pause_df['is_pause_hat'] = is_pause_after_hat
    pause_df['pause_duration_hat'] = pause_duration_hat

    print('Calculate metrics, training fold; Exclude last tokens in every sentence!')
    calc_metrics(pause_df[(pause_df.set == 'train') & (pause_df.is_last_word == 0)])

    print('\nCalculate metrics, testing fold; Exclude last tokens in every sentence!')
    calc_metrics(pause_df[(pause_df.set == 'test') & (pause_df.is_last_word == 0)])


if __name__ == '__main__':
    test_pause_predictor()
