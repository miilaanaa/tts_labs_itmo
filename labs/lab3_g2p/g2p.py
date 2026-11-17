"""Grapheme-to-phoneme front end — lab 3.

Run as a script to score the module on the prepared data::

    python g2p.py
"""
import csv
import os
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import tqdm

TOKENS_PHONES_DATA = 'data/tokens_phones.csv'
MFA_DICT_PATH = 'data/russian_mfa.dict'

SIL = '<SIL>'

# Наивный графемо-фонемный маппинг для OOV-слов (последний резерв).
GRAPHEME_TO_PHONEME = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'ɡ', 'д': 'd̪', 'е': 'e', 'ё': 'o',
    'ж': 'ʐ', 'з': 'z̪', 'и': 'i', 'й': 'j', 'к': 'k', 'л': 'ɫ', 'м': 'm',
    'н': 'n̪', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's̪', 'т': 't̪', 'у': 'u',
    'ф': 'f', 'х': 'x', 'ц': 't̪s̪', 'ч': 'tɕ', 'ш': 'ʂ', 'щ': 'ɕː',
    'ы': 'ɨ', 'ь': '', 'ъ': '', 'э': 'ɛ', 'ю': 'u', 'я': 'a',
}

# Согласные и их мягкие пары (для смягчения перед е, ё, и, ю, я, ь).
SOFT_PAIRS = {
    'b': 'bʲ', 'v': 'vʲ', 'ɡ': 'ɟ', 'd̪': 'dʲ', 'z̪': 'zʲ', 'k': 'c',
    'ɫ': 'ʎ', 'm': 'mʲ', 'n̪': 'ɲ', 'p': 'pʲ', 'r': 'rʲ', 's̪': 'sʲ',
    't̪': 'tʲ', 'f': 'fʲ', 'x': 'ç',
}

# Оглушение согласных в конце слова.
DEVOICE = {
    'b': 'p', 'bʲ': 'pʲ', 'v': 'f', 'vʲ': 'fʲ', 'ɡ': 'k', 'ɟ': 'c',
    'd̪': 't̪', 'dʲ': 'tʲ', 'z̪': 's̪', 'zʲ': 'sʲ', 'ʐ': 'ʂ',
}

# Гласные переднего ряда, вызывающие смягчение предыдущего согласного.
FRONT_VOWELS = set('еёиюя')


def load_mfa_dict(path: str = MFA_DICT_PATH) -> dict[str, str]:
    """Load the plain MFA dictionary: word -> first pronunciation (space-separated)."""
    result: dict[str, str] = {}
    if not os.path.exists(path):
        print(f'MFA dictionary not found at {path}, skipping')
        return result
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 2:
                continue
            word, phones = parts
            if word not in result:
                result[word] = phones
    return result


class G2P:
    """Turns a sentence into a phone sequence."""

    def __init__(self,
                 lexicon: dict[str, Counter] | None = None,
                 mfa_dict: dict[str, str] | None = None) -> None:
        self.lexicon = lexicon or {}
        self.mfa_dict = mfa_dict or {}

    def transcribe(self, tokens: list[str] | np.ndarray) -> list[list[str]]:
        """Transcribe each token of one sentence."""
        result: list[list[str]] = []
        for token in tokens:
            if token == SIL:
                result.append([SIL])
                continue
            # 1. train-лексикон: самое частое произношение
            if token in self.lexicon:
                best = self.lexicon[token].most_common(1)[0][0]
                result.append(best.split() if best else [])
                continue
            # 2. MFA-словарь: первое произношение
            if token in self.mfa_dict:
                result.append(self.mfa_dict[token].split())
                continue
            # 3. наивный маппинг
            result.append(self._naive_oov(token))
        return result

    def _naive_oov(self, word: str) -> list[str]:
        """Наивный графемо-фонемный маппинг для OOV-слова."""
        phones: list[str] = []
        word = word.lower()
        for ch in word:
            if ch == 'ь':
                if phones:
                    phones[-1] = SOFT_PAIRS.get(phones[-1], phones[-1] + 'ʲ')
                continue
            if ch == 'ъ':
                continue
            ph = GRAPHEME_TO_PHONEME.get(ch)
            if ph is None:
                continue
            # смягчение предыдущего согласного перед гласными переднего ряда
            if ch in FRONT_VOWELS and phones:
                phones[-1] = SOFT_PAIRS.get(phones[-1], phones[-1])
            phones.append(ph)
        # оглушение в конце слова
        if phones and phones[-1] in DEVOICE:
            phones[-1] = DEVOICE[phones[-1]]
        return phones


def phone_error_rate(reference: list[list[str]], hypothesis: list[list[str]]) -> float:
    """Corpus-level PER: summed edit distance over summed reference length."""
    total_distance = 0
    total_length = 0
    for ref, hyp in zip(reference, hypothesis):
        total_length += len(ref)
        if not hyp:
            total_distance += len(ref)
            continue
        previous = list(range(len(hyp) + 1))
        for i, ref_phone in enumerate(ref, start=1):
            current = [i] + [0] * len(hyp)
            for j, hyp_phone in enumerate(hyp, start=1):
                current[j] = min(previous[j] + 1,
                                 current[j - 1] + 1,
                                 previous[j - 1] + (ref_phone != hyp_phone))
            previous = current
        total_distance += previous[-1]
    return total_distance / total_length if total_length else float('nan')


def word_accuracy(reference: list[list[str]], hypothesis: list[list[str]]) -> float:
    """Fraction of words whose whole phone sequence is exactly right."""
    if not reference:
        return float('nan')
    return float(np.mean([ref == hyp for ref, hyp in zip(reference, hypothesis)]))


def load_tokens(path: str = TOKENS_PHONES_DATA) -> pd.DataFrame:
    """Read the training table and drop tokens with no reference transcription."""
    tokens = pd.read_csv(path, sep='|', quoting=csv.QUOTE_NONE)
    tokens['token'] = tokens.token.astype(str).map(lambda s: unicodedata.normalize('NFC', s))
    tokens['phones'] = tokens.phones.fillna('').astype(str)

    unpronounced = (tokens.phones == '') | tokens.phones.str.split().map(lambda p: 'spn' in p)
    print(f'dropping {unpronounced.sum()} tokens without a transcription '
          f'({unpronounced.mean():.2%})')
    return tokens[~unpronounced].reset_index(drop=True)


def build_lexicon(tokens: pd.DataFrame) -> dict[str, Counter]:
    """Count pronunciations per token type over the given rows."""
    lexicon: dict[str, Counter] = defaultdict(Counter)
    for token, phones in tokens[['token', 'phones']].values:
        lexicon[token][phones] += 1
    return lexicon


def calc_metrics(g2p: G2P, tokens: pd.DataFrame, lexicon: dict[str, Counter]) -> None:
    """Print running-text PER, and PER and word accuracy on out-of-vocabulary tokens."""
    sentence_ref: list[list[str]] = []
    sentence_hyp: list[list[str]] = []
    oov_ref: list[list[str]] = []
    oov_hyp: list[list[str]] = []
    in_vocab_ref: list[list[str]] = []
    in_vocab_hyp: list[list[str]] = []

    for _, sentence in tqdm.tqdm(tokens.groupby('id', sort=False)):
        sentence = sentence.sort_values('token_id')
        words = sentence.token.values
        predicted = g2p.transcribe(words)
        if len(predicted) != len(words):
            raise ValueError(f'transcribe returned {len(predicted)} outputs '
                             f'for {len(words)} tokens')
        reference = [p.split() for p in sentence.phones.values]

        sentence_ref.append([p for word in reference for p in word if p != SIL])
        sentence_hyp.append([p for word in predicted for p in word if p != SIL])
        for token, ref, hyp in zip(words, reference, predicted):
            if token in lexicon:
                in_vocab_ref.append(ref)
                in_vocab_hyp.append(hyp)
            else:
                oov_ref.append(ref)
                oov_hyp.append(hyp)

    n_words = len(in_vocab_ref) + len(oov_ref)
    print(f'running text  PER {phone_error_rate(sentence_ref, sentence_hyp):.4f}')
    print(f'in-vocabulary PER {phone_error_rate(in_vocab_ref, in_vocab_hyp):.4f}  '
          f'({len(in_vocab_ref)} words, {len(in_vocab_ref) / n_words:.1%})')
    print(f'OOV           PER {phone_error_rate(oov_ref, oov_hyp):.4f}  '
          f'word accuracy {word_accuracy(oov_ref, oov_hyp):.4f}  '
          f'({len(oov_ref)} words, {len(oov_ref) / n_words:.1%})')


def test_g2p() -> None:
    """Build the lexicon from `train` and score the module on both folds."""
    tokens = load_tokens()
    train = tokens[tokens.set == 'train']
    test = tokens[tokens.set == 'test']
    print(f'train {len(train)} tokens / test {len(test)} tokens')

    lexicon = build_lexicon(train)
    print(f'lexicon: {len(lexicon)} types, '
          f'{sum(1 for c in lexicon.values() if len(c) > 1)} with several pronunciations')

    mfa_dict = load_mfa_dict()
    print(f'MFA dictionary: {len(mfa_dict)} words')

    g2p = G2P(lexicon, mfa_dict)

    print('\nCalculate metrics, training fold:')
    calc_metrics(g2p, train, lexicon)

    print('\nCalculate metrics, testing fold:')
    calc_metrics(g2p, test, lexicon)


if __name__ == '__main__':
    test_g2p()
