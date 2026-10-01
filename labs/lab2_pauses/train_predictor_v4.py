"""Обучение CatBoost v4 с богатыми фичами (POS, next_punct_class, prev/next word)."""
import csv
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import f1_score, precision_score, recall_score, mean_absolute_error

from features_v4 import build_features_v4, CAT_FEATURES

HERE = Path(__file__).resolve().parent
META_PATH = HERE / 'data' / 'RUSLAN_pause_metadata.csv'
CLF_PATH = HERE / 'data' / 'pause_clf_v4.pkl'
REG_PATH = HERE / 'data' / 'pause_reg_v4.pkl'
FEAT_CACHE = HERE / 'data' / 'features_v4_cache.pkl'


def split_train_val(df, val_frac=0.2, seed=42):
    train_ids = df[df['set'] == 'train']['id'].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(train_ids)
    n_val = int(len(train_ids) * val_frac)
    val_ids = set(train_ids[:n_val])
    is_val = df['id'].isin(val_ids).values
    is_train = (df['set'] == 'train').values & ~is_val
    is_test = (df['set'] == 'test').values
    return is_train, is_val, is_test


def main():
    df = pd.read_csv(META_PATH, sep='|', quoting=csv.QUOTE_NONE)
    print(f'Всего строк: {len(df)}')

    # Признаки кешируются, чтобы не считать POS-теги повторно
    if FEAT_CACHE.exists():
        print(f'Читаю кеш признаков: {FEAT_CACHE}')
        feats = pd.read_pickle(FEAT_CACHE)
    else:
        print('Строю признаки (pymorphy3, ~1-2 минуты)...')
        feats = build_features_v4(df)
        feats.to_pickle(FEAT_CACHE)
        print(f'Кеш сохранён: {FEAT_CACHE}')

    y_clf = df['is_pause_after'].values.astype(int)
    y_reg = df['pause_duration'].values.astype(float)

    is_train, is_val, is_test = split_train_val(df)
    last_zero = (df['is_last_word'] == 0).values

    mask_train = is_train & last_zero
    mask_val = is_val & last_zero
    mask_test = is_test & last_zero

    X_train = feats[mask_train]
    y_train = y_clf[mask_train]
    X_val = feats[mask_val]
    y_val = y_clf[mask_val]
    X_test = feats[mask_test]
    y_test = y_clf[mask_test]
    print(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')

    train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
    val_pool = Pool(X_val, y_val, cat_features=CAT_FEATURES)
    test_pool = Pool(X_test, y_test, cat_features=CAT_FEATURES)

    print('\nОбучаем CatBoostClassifier (v4)...')
    clf = CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.07,
        l2_leaf_reg=3.0,
        loss_function='Logloss',
        eval_metric='F1',
        class_weights=[1.0, 2.0],
        random_seed=42,
        verbose=100,
        od_type='Iter',
        od_wait=80,
    )
    clf.fit(train_pool, eval_set=val_pool, use_best_model=True)

    proba_val = clf.predict_proba(val_pool)[:, 1]
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.20, 0.71, 0.01):
        y_hat = (proba_val > thr).astype(int)
        f1 = f1_score(y_val, y_hat)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    print(f'>>> Лучший порог: {best_thr:.2f}, val F1={best_f1:.4f}')

    proba_train = clf.predict_proba(train_pool)[:, 1]
    proba_test = clf.predict_proba(test_pool)[:, 1]

    print('\nМетрики классификатора:')
    for name, y_true, proba in [
        ('train', y_train, proba_train),
        ('val',   y_val,   proba_val),
        ('test',  y_test,  proba_test),
    ]:
        y_hat = (proba > best_thr).astype(int)
        print(f'  {name:5s}: PRC={precision_score(y_true, y_hat):.4f}  '
              f'REC={recall_score(y_true, y_hat):.4f}  '
              f'F1={f1_score(y_true, y_hat):.4f}')

    # --- Регрессор ---
    print('\nОбучаем CatBoostRegressor (v4)...')
    reg_mask_train = mask_train & (y_clf == 1)
    Xreg_train = feats[reg_mask_train]
    yreg_train = y_reg[reg_mask_train]
    reg_train_pool = Pool(Xreg_train, yreg_train, cat_features=CAT_FEATURES)

    reg = CatBoostRegressor(
        iterations=800, depth=6, learning_rate=0.07, l2_leaf_reg=3.0,
        loss_function='MAE', random_seed=42, verbose=100,
    )
    reg.fit(reg_train_pool)

    for name, mask, proba in [('val', mask_val, proba_val), ('test', mask_test, proba_test)]:
        y_true_sub = y_clf[mask]
        tp_local = (y_true_sub == 1) & (proba > best_thr)
        if tp_local.sum() == 0:
            print(f'  MAE {name}: n/a')
            continue
        Xtp = feats[mask][tp_local]
        yhat = reg.predict(Xtp)
        mae = mean_absolute_error(y_reg[mask][tp_local], yhat)
        print(f'  MAE {name}: {mae:.4f}  (TP={tp_local.sum()})')

    with open(CLF_PATH, 'wb') as f:
        pickle.dump({'model': clf, 'threshold': float(best_thr)}, f)
    with open(REG_PATH, 'wb') as f:
        pickle.dump(reg, f)
    print(f'\nСохранено: {CLF_PATH} (thr={best_thr:.2f})')
    print(f'Сохранено: {REG_PATH}')


if __name__ == '__main__':
    main()
