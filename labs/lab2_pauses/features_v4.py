"""Построение признаков для предиктора пауз — v4 (rich features + POS).

Используется одинаково при обучении (train_predictor_v4.py) и при инференсе
(pause_predictor.py).

Оптимизации:
  - pymorphy3-разбор мемоизируется по слову (иначе — сотни тысяч лишних вызовов)
  - линейные проходы вместо O(n^2) для tokens_to_next_comma / next_punct_class
  - commas_left_in_clause считается одним проходом с конца
"""
import re
import numpy as np
import pandas as pd
import pymorphy3

_MORPH = pymorphy3.MorphAnalyzer()
VOWELS = set('аеёиоуыэюя')

PUNCT_TO_CLASS = {
    '.': 'period', ',': 'comma', '?': 'question', '!': 'exclaim',
    '—': 'dash', '–': 'dash', '-': 'dash', '…': 'ellipsis',
    ';': 'semicolon', ':': 'colon', '»': 'closing', '«': 'closing',
    ')': 'closing', '"': 'other',
}

TRAILING_PUNCT_RE = re.compile(r'([.,!?;:…—–\-»«")]+)$')


def extract_punct(token) -> str:
    m = TRAILING_PUNCT_RE.search(str(token))
    if not m:
        return ''
    for ch in m.group(1):
        if ch in PUNCT_TO_CLASS:
            return ch
    return ''


def strip_punct(token) -> str:
    return TRAILING_PUNCT_RE.sub('', str(token))


def punct_class(token) -> str:
    p = extract_punct(token)
    if not p:
        return 'none'
    return PUNCT_TO_CLASS.get(p, 'other')


def word_len_clean(token) -> int:
    return len(strip_punct(token))


_SYL_CACHE: dict = {}

def count_syllables(word: str) -> int:
    cached = _SYL_CACHE.get(word)
    if cached is not None:
        return cached
    n = sum(1 for c in word.lower() if c in VOWELS)
    _SYL_CACHE[word] = n
    return n


_POS_CACHE: dict = {}

def pos_tag(word: str) -> str:
    """POS-тег через pymorphy3 (первый разбор). Мемоизация по слову."""
    w = strip_punct(word).lower()
    if not w or not w.isalpha():
        return 'NONE'
    cached = _POS_CACHE.get(w)
    if cached is not None:
        return cached
    tag = str(_MORPH.parse(w)[0].tag.POS or 'NONE')
    _POS_CACHE[w] = tag
    return tag


def build_features_v4(df: pd.DataFrame) -> pd.DataFrame:
    """Строит датафрейм признаков в том же порядке, что и df."""
    n = len(df)
    tokens = df['label_raw'].astype(str).values
    ids = df['id'].values

    # --- Позиция в предложении, длина предложения, число предложений в id ---
    pos_in_sent = np.zeros(n, dtype=int)
    sent_len = np.zeros(n, dtype=int)

    id_sent_count = {}
    cur_id = None
    cnt = 0
    for _id in ids:
        if _id != cur_id:
            if cur_id is not None:
                id_sent_count[cur_id] = cnt
            cur_id = _id
            cnt = 1
        else:
            cnt += 1
    if cur_id is not None:
        id_sent_count[cur_id] = cnt

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

    n_sentences_in_id = np.array([id_sent_count.get(_id, 1) for _id in ids], dtype=int)

    # --- Пословные фичи (мемоизация) ---
    puncts = [extract_punct(t) for t in tokens]
    punct_classes = [PUNCT_TO_CLASS.get(p, 'none') if p else 'none' for p in puncts]
    word_lens = [word_len_clean(t) for t in tokens]
    syllables = [count_syllables(strip_punct(t)) for t in tokens]
    poses = [pos_tag(t) for t in tokens]

    # --- Проходные фичи (без O(n^2)) ---
    next_punct_class = ['<PAD>'] * n
    last_punct_class = '<NONE>'
    for i in range(n - 1, -1, -1):
        if i == n - 1 or ids[i] != ids[i + 1]:
            last_punct_class = '<NONE>'
        if puncts[i]:
            last_punct_class = punct_classes[i]
        next_punct_class[i] = last_punct_class

    tokens_to_next_comma = np.full(n, 999, dtype=int)
    last_comma = -1
    for i in range(n - 1, -1, -1):
        if i == n - 1 or ids[i] != ids[i + 1]:
            last_comma = -1
        if puncts[i] == ',':
            last_comma = i
        if last_comma >= 0 and last_comma > i:
            tokens_to_next_comma[i] = last_comma - i

    commas_left_in_clause = np.zeros(n, dtype=int)
    remaining = 0
    prev_id = None
    for i in range(n - 1, -1, -1):
        if ids[i] != prev_id:
            remaining = 0
            prev_id = ids[i]
        if puncts[i] == ',':
            remaining += 1
        commas_left_in_clause[i] = remaining

    tokens_since_punct = np.zeros(n, dtype=int)
    for i in range(n):
        if i == 0 or ids[i] != ids[i - 1]:
            tokens_since_punct[i] = 0
        else:
            tokens_since_punct[i] = tokens_since_punct[i - 1] + 1
        if puncts[i]:
            tokens_since_punct[i] = 0

    tokens_to_punct = np.zeros(n, dtype=int)
    for i in range(n - 1, -1, -1):
        if i == n - 1 or ids[i] != ids[i + 1]:
            tokens_to_punct[i] = 0
        else:
            tokens_to_punct[i] = tokens_to_punct[i + 1] + 1
        if puncts[i]:
            tokens_to_punct[i] = 0

    tokens_to_end = sent_len - pos_in_sent - 1

    # --- Соседние слова и POS ---
    prev_word = ['<PAD>'] * n
    next_word = ['<PAD>'] * n
    pos_prev1 = ['NONE'] * n
    pos_prev2 = ['NONE'] * n
    pos_next1 = ['NONE'] * n
    pos_next2 = ['NONE'] * n
    for i in range(n):
        if i > 0 and ids[i] == ids[i - 1]:
            prev_word[i] = strip_punct(tokens[i - 1]).lower()
            pos_prev1[i] = poses[i - 1]
            if i > 1 and ids[i] == ids[i - 2]:
                pos_prev2[i] = poses[i - 2]
        if i < n - 1 and ids[i] == ids[i + 1]:
            next_word[i] = strip_punct(tokens[i + 1]).lower()
            pos_next1[i] = poses[i + 1]
            if i < n - 2 and ids[i] == ids[i + 2]:
                pos_next2[i] = poses[i + 2]

    has_quote_mark = np.array(
        [int('"' in str(t) or '«' in str(t) or '»' in str(t)) for t in tokens],
        dtype=int,
    )

    sl = np.where(sent_len > 0, sent_len, 1)
    rel_pos = pos_in_sent / sl

    feats = pd.DataFrame({
        'punct_class': punct_classes,
        'next_punct_class': next_punct_class,
        'has_quote_mark': has_quote_mark,
        'word_len': word_lens,
        'n_syllables': syllables,
        'prev_word': prev_word,
        'next_word': next_word,
        'pos_curr': poses,
        'pos_prev1': pos_prev1,
        'pos_prev2': pos_prev2,
        'pos_next1': pos_next1,
        'pos_next2': pos_next2,
        'pos_in_sentence': pos_in_sent,
        'sent_len': sent_len,
        'rel_pos': rel_pos,
        'tokens_to_end': tokens_to_end,
        'n_sentences_in_id': n_sentences_in_id,
        'tokens_since_punct': tokens_since_punct,
        'tokens_to_punct': tokens_to_punct,
        'tokens_to_next_comma': tokens_to_next_comma,
        'commas_left_in_clause': commas_left_in_clause,
    })
    return feats


CAT_FEATURES = [
    'punct_class', 'next_punct_class',
    'pos_curr', 'pos_prev1', 'pos_prev2', 'pos_next1', 'pos_next2',
    'prev_word', 'next_word',
]
