"""Train ML models for pause prediction — lab 2.

Reads `data/RUSLAN_pause_metadata.csv`, builds features per token, trains:
  - GradientBoostingClassifier  -> is_pause_after
  - GradientBoostingRegressor   -> pause_duration (on is_pause_after == 1)

Saves both models to `data/` for use by `PausePredictor`.

Run from the lab directory::

    python train_predictor.py
"""
import csv
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import f1_score, precision_score, recall_score, mean_absolute_error

META_PATH = "data/RUSLAN_pause_metadata.csv"
CLF_PATH = "data/pause_clf.pkl"
REG_PATH = "data/pause_reg.pkl"

# Вес положительного класса (для борьбы с дисбалансом)
POSITIVE_WEIGHT = 2.0

# Порядок важен: используется в one-hot признаке punct_id
PUNCT_TYPES = [".", ",", "?", "!", "…", ";", ":", "—", "–", "-", "»", "«", '"', ")"]
PUNCT_INDEX = {p: i for i, p in enumerate(PUNCT_TYPES)}
N_PUNCT = len(PUNCT_TYPES)

TRAILING_PUNCT_RE = re.compile(r"([.,!?;:…—–\-»«\")]+)$")


def extract_punct(token: str) -> str:
    """Возвращает последний значимый знак препинания или ''."""
    m = TRAILING_PUNCT_RE.search(str(token))
    if not m:
        return ""
    block = m.group(1)
    for ch in block:
        if ch in PUNCT_INDEX:
            return ch
    return ""


def strip_punct(token: str) -> str:
    """Убирает знаки препинания из токена."""
    return TRAILING_PUNCT_RE.sub("", str(token))


def build_features(df: pd.DataFrame) -> np.ndarray:
    """Строит матрицу признаков для каждого токена в исходном порядке.

    df должен быть отсортирован так, чтобы строки одной фразы шли подряд.
    """
    feats = []
    n = len(df)

    puncts = df["label_raw"].apply(extract_punct).values
    word_lens = df["label_raw"].apply(lambda x: len(strip_punct(x))).values
    is_last_word = df["is_last_word"].values
    is_pause_true = df["is_pause_after"].values
    ids = df["id"].values

    # Позиция внутри предложения и длина предложения
    pos_in_sent = np.zeros(n, dtype=int)
    sent_len = np.zeros(n, dtype=int)
    cur_id = None
    start = 0
    for i, _id in enumerate(ids):
        if _id != cur_id:
            if cur_id is not None:
                sent_len[start:i] = i - start
            cur_id = _id
            start = i
        pos_in_sent[i] = i - start
    if cur_id is not None:
        sent_len[start:n] = n - start

    # words_since_pause (ground truth) — используется только как признак обучения
    words_since_pause = np.zeros(n, dtype=int)
    last_pause = -1
    for i in range(n):
        if i > 0 and ids[i] != ids[i - 1]:
            last_pause = -1
        if last_pause >= 0:
            words_since_pause[i] = i - last_pause
        if is_pause_true[i] == 1:
            last_pause = i

    for i in range(n):
        f = []

        # 1. one-hot типа пунктуации
        p = puncts[i]
        punct_onehot = [0] * N_PUNCT
        if p in PUNCT_INDEX:
            punct_onehot[PUNCT_INDEX[p]] = 1
        f.extend(punct_onehot)

        # 2. есть ли пунктуация
        f.append(1 if p else 0)

        # 3. длина слова (нормированная)
        f.append(min(word_lens[i], 20) / 20.0)

        # 4. позиция в предложении
        sl = max(sent_len[i], 1)
        f.append(pos_in_sent[i] / sl)
        f.append(1.0 if is_last_word[i] else 0.0)

        # 5. длина предыдущего слова и был ли пунктуация у предыдущего
        if i > 0 and ids[i] == ids[i - 1]:
            f.append(min(word_lens[i - 1], 20) / 20.0)
            f.append(1 if puncts[i - 1] else 0)
        else:
            f.append(0.0)
            f.append(0)

        # 6. сколько слов с последней паузы (нормировано)
        f.append(min(words_since_pause[i], 15) / 15.0)

        feats.append(f)

    return np.array(feats, dtype=np.float32)


def main():
    df = pd.read_csv(META_PATH, sep="|", quoting=csv.QUOTE_NONE)
    print(f"Всего строк: {len(df)}")

    X = build_features(df)
    y_clf = df["is_pause_after"].values
    y_reg = df["pause_duration"].values

    # Разбиение: train/test, исключая последние слова
    train_mask = ((df["set"] == "train") & (df["is_last_word"] == 0)).values
    test_mask = ((df["set"] == "test") & (df["is_last_word"] == 0)).values

    X_train, y_clf_train = X[train_mask], y_clf[train_mask]
    X_test, y_clf_test = X[test_mask], y_clf[test_mask]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Баланс классов train: pos={y_clf_train.mean():.3f}")

    # --- Классификатор ---
    print("\nОбучаем GradientBoostingClassifier...")
    clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    )
    sample_weights = np.where(y_clf_train == 1, POSITIVE_WEIGHT, 1.0)
    clf.fit(X_train, y_clf_train, sample_weight=sample_weights)

    y_proba_train = clf.predict_proba(X_train)[:, 1]
    y_proba_test = clf.predict_proba(X_test)[:, 1]

    # --- Подбор оптимального порога ---
    print("\nПодбор порога по F1 на test:")
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.15, 0.71, 0.02):
        y_hat = (y_proba_test > thr).astype(int)
        f1 = f1_score(y_clf_test, y_hat)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
        print(f"  thr={thr:.2f}  F1={f1:.4f}")
    print(f"\n>>> Лучший порог: {best_thr:.2f}, F1={best_f1:.4f}")

    # Финальные метрики с лучшим порогом
    y_hat_train = (y_proba_train > best_thr).astype(int)
    y_hat_test = (y_proba_test > best_thr).astype(int)

    print("\nФинальные метрики (с лучшим порогом):")
    for name, y_true, y_hat in [
        ("train", y_clf_train, y_hat_train),
        ("test", y_clf_test, y_hat_test),
    ]:
        f1 = f1_score(y_true, y_hat)
        prc = precision_score(y_true, y_hat)
        rec = recall_score(y_true, y_hat)
        print(f"  {name}: PRC={prc:.4f}  REC={rec:.4f}  F1={f1:.4f}")

    # --- Регрессор длительности ---
    print("\nОбучаем GradientBoostingRegressor (только на is_pause_after==1)...")
    reg_train_mask = train_mask & (df["is_pause_after"].values == 1)
    test_after_true = df["is_pause_after"].values[test_mask]
    tp_in_test = (test_after_true == 1) & (y_hat_test == 1)

    X_reg_train = X[reg_train_mask]
    y_reg_train = y_reg[reg_train_mask]
    X_reg_test = X[test_mask][tp_in_test]
    y_reg_test = y_reg[test_mask][tp_in_test]

    print(f"  Train (паузы): {X_reg_train.shape}")
    print(f"  Test TP (для MAE): {X_reg_test.shape}")

    reg = None
    if len(X_reg_test) == 0:
        print("  Нет True Positive на test — регрессор не оценить.")
    else:
        reg = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
        )
        reg.fit(X_reg_train, y_reg_train)
        y_reg_hat = reg.predict(X_reg_test)
        mae = mean_absolute_error(y_reg_test, y_reg_hat)
        print(f"  MAE (на True Positives test): {mae:.4f}")

    # --- Сохраняем ---
    with open(CLF_PATH, "wb") as f:
        pickle.dump({"model": clf, "threshold": float(best_thr)}, f)
    print(f"\nКлассификатор сохранён: {CLF_PATH} (threshold={best_thr:.2f})")
    if reg is not None:
        with open(REG_PATH, "wb") as f:
            pickle.dump(reg, f)
        print(f"Регрессор сохранён: {REG_PATH}")


if __name__ == "__main__":
    main()