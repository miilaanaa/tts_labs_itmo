"""Grapheme-to-phoneme front end — lab 3.

Run as a script to score the module on the prepared data::

    python g2p.py

Two phone error rates are reported on the split carried in the data's own `set` column: PER over
whole `test` utterances — the front end as lab 5 will run it — and PER over `test` tokens
absent from the training lexicon, which is the out-of-vocabulary model alone.
"""
import csv
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import tqdm

TOKENS_PHONES_DATA = 'data/tokens_phones.csv'
MFA_DICT_PATH = 'data/russian_mfa.dict'


SIL = '<SIL>'


# ---- Правила графемно-фонемного преобразования для OOV ----

LETTER_TO_PHONE = {
    'а': 'a',   'б': 'b',   'в': 'v',   'г': 'ɡ',   'д': 'd̪',
    'е': 'e',   'ё': 'o',   'ж': 'ʐ',   'з': 'z̪',   'и': 'i',
    'й': 'j',   'к': 'k',   'л': 'ɫ',   'м': 'm',   'н': 'n̪',
    'о': 'o',   'п': 'p',   'р': 'r',   'с': 's̪',   'т': 't̪',
    'у': 'u',   'ф': 'f',   'х': 'x',   'ц': 't̪s̪', 'ч': 'tɕ',
    'ш': 'ʂ',   'щ': 'ɕ',   'ы': 'ɨ',   'э': 'ɛ',   'ю': 'u',
    'я': 'a',
}

SOFT_CONSONANTS = {
    'б': 'bʲ', 'в': 'vʲ', 'д': 'dʲ', 'з': 'zʲ', 'к': 'c',
    'л': 'ʎ',  'м': 'mʲ', 'н': 'ɲ',  'п': 'pʲ', 'р': 'rʲ',
    'с': 'sʲ', 'т': 'tʲ', 'ф': 'fʲ', 'г': 'ɟ',  'х': 'ç',
}

FINAL_DEVOICING = {
    'b': 'p',    'v': 'f',    'ɡ': 'k',    'd̪': 't̪',  'z̪': 's̪',
    'ʐ': 'ʂ',    'bʲ': 'pʲ',  'vʲ': 'fʲ',  'dʲ': 'tʲ', 'zʲ': 'sʲ',
    'ɟ': 'c',
}

VOICELESS_PHONES = {
    'p', 'pʲ', 'f', 'fʲ', 't̪', 'tʲ', 's̪', 'sʲ',
    'k', 'c', 'ʂ', 'ɕ', 'tɕ', 't̪s̪', 'x', 'ç', 'tʂ',
}

CONSONANT_DEVOICING = {
    'b': 'p',    'bʲ': 'pʲ',
    'v': 'f',    'vʲ': 'fʲ',
    'ɡ': 'k',    'ɟ': 'c',
    'd̪': 't̪',  'dʲ': 'tʲ',
    'z̪': 's̪',  'zʲ': 'sʲ',
    'ʐ': 'ʂ',    'dʐ': 'tʂ',
}

SILENT_CLUSTER_RULES = (
    ('стн', 'сн'),
    ('здн', 'зн'),
    ('ндц', 'нц'),
    ('рдц', 'рц'),
    ('лнц', 'нц'),
)

SIBILANT_CLUSTER_RULES = (
    ('сч', 'щ'),
    ('зч', 'щ'),
    ('жч', 'щ'),
)

RUSSIAN_VOWELS = set('аеёиоуыэюя')

VOWEL_PHONES = {'a', 'o', 'e', 'i', 'u', 'ɨ', 'ɛ', 'ɐ', 'ə', 'ɪ', 'ʊ', 'ʉ'}

REDUCED_VOWELS = {'ɐ', 'ə', 'ɪ', 'ʊ'}


# ---- Класс G2P ----

class G2P:
    """Turns a sentence into a phone sequence.

    Args:
        lexicon: `word -> Counter({phones: count})` as written by `prepare_lexicon.py`.
            Pronunciations are space-separated NFC strings. Keys are lowercase and carry no
            stress mark, so look words up accordingly.
        mfa_dict: `word -> phones` for the Russian MFA pronunciation dictionary.
            Used as a fallback when a token is not in `lexicon`.
    """

    def __init__(self, lexicon: dict[str, Counter] | None = None,
                 mfa_dict: dict[str, str] | None = None) -> None:
        self.lexicon = lexicon or {}
        self.mfa_dict = mfa_dict or {}

        # Индекс ударений: word -> номер ударного слога + префиксный индекс
        # для быстрого поиска похожих слов
        self.stress_examples: dict[str, int] = {}
        self.stress_index: dict[str, list[str]] = defaultdict(list)
        for word, pronunciations in self.lexicon.items():
            phones = pronunciations.most_common(1)[0][0].split()
            stress = self.infer_stress_from_phones(word, phones)
            if stress is not None:
                self.stress_examples[word] = stress
                self.stress_index[word[:4]].append(word)

    # ---- Утилиты ----

    def normalize(self, token: str) -> str:
        return unicodedata.normalize('NFC', token.lower().strip())

    def infer_stress_from_phones(self, word: str, phones: list[str]) -> int | None:
        """Оценка индекса ударного слога по фонемам train-слова.

        Идея: в русском безударные гласные редуцированы (ɐ, ə, ɪ, ʊ),
        ударные — полные (a, o, e, i, u, ы...). Если в слове ровно
        одна «полная» гласная — она и есть ударная.
        """
        letters = [c for c in word if c in RUSSIAN_VOWELS]
        sounds = [p for p in phones if p in VOWEL_PHONES]
        if not letters or len(letters) != len(sounds):
            return None
        if 'ё' in letters:
            return letters.index('ё')
        candidates = [i for i, p in enumerate(sounds) if p not in REDUCED_VOWELS]
        return candidates[0] if len(candidates) == 1 else None

    def predict_stress(self, word: str) -> int | None:
        """Перенос ударения с похожего train-слова (по первым 4 буквам)."""
        letters = [c for c in word if c in RUSSIAN_VOWELS]
        if 'ё' in letters:
            return letters.index('ё')
        if len(word) < 5:
            return None
        best_score, best_stress = 0.0, None
        for candidate in self.stress_index.get(word[:4], []):
            if sum(c in RUSSIAN_VOWELS for c in candidate) != len(letters):
                continue
            score = SequenceMatcher(None, word, candidate).ratio()
            if score > best_score:
                best_score, best_stress = score, self.stress_examples[candidate]
        return best_stress if best_score >= 0.80 else None

    # ---- Подготовка орфографии ----

    def normalize_oov_endings(self, word: str) -> str:
        """Правило -ого/-его → -ово/-ево (по мотивам русских прилагательных)."""
        exceptions = {'много', 'немного', 'строго', 'нестрого', 'итого'}
        if word in exceptions:
            return word
        if word.endswith('ого'):
            return word[:-2] + 'во'
        if word.endswith('его'):
            return word[:-2] + 'во'
        return word

    def prepare_oov_spelling(self, word: str) -> str:
        """Орфографические преобразования до фонемного чтения."""
        word = self.normalize(word)
        word = self.normalize_oov_endings(word)
        for old, new in SILENT_CLUSTER_RULES:
            word = word.replace(old, new)
        for old, new in SIBILANT_CLUSTER_RULES:
            word = word.replace(old, new)
        return word

    # ---- Оглушение ----

    def devoice_consonants(self, phones: list[str]) -> list[str]:
        """Регрессивное оглушение + оглушение на конце слова."""
        result = phones.copy()
        for i in range(len(result) - 2, -1, -1):
            if (result[i] in CONSONANT_DEVOICING
                    and result[i + 1] in VOICELESS_PHONES):
                result[i] = CONSONANT_DEVOICING[result[i]]
        if result and result[-1] in FINAL_DEVOICING:
            result[-1] = FINAL_DEVOICING[result[-1]]
        return result

    # ---- Основной OOV-модуль ----

    def transcribe_oov(self, word: str) -> list[str]:
        """Правиловая транскрипция OOV с редукцией гласных и ударением."""
        original_word = self.normalize(word)
        word = self.prepare_oov_spelling(original_word)

        # Определяем ударный слог
        vowel_positions = [i for i, c in enumerate(word) if c in RUSSIAN_VOWELS]
        vowel_to_syllable = {pos: n for n, pos in enumerate(vowel_positions)}
        predicted_stress = self.predict_stress(original_word)
        if predicted_stress is not None and 0 <= predicted_stress < len(vowel_positions):
            stressed_syllable = predicted_stress
        elif vowel_positions:
            # Эвристика: предпоследний слог
            stressed_syllable = max(0, len(vowel_positions) - 2)
        else:
            stressed_syllable = -1

        def vowel_phone(char: str, pos: int, base: str, preceding: str) -> str:
            """Фонема для гласной буквы с учётом ударности и редукции."""
            syllable = vowel_to_syllable[pos]
            stressed = syllable == stressed_syllable
            if stressed:
                # Безударный «и» после ж/ш/ц произносится как ɨ
                if char == 'и' and preceding in 'жшц':
                    return 'ɨ'
                return base
            # Безударные позиции — редукция
            if char in {'о', 'а'}:
                if syllable == stressed_syllable - 1:
                    # 1-й предударный слог → ɐ
                    return 'ɐ'
                if pos == 0 and syllable < stressed_syllable:
                    # Абсолютное начало слова перед ударением → ɐ
                    return 'ɐ'
                return 'ə'
            if char in {'е', 'я', 'и'}:
                if char == 'и' and preceding in 'жшц':
                    return 'ɨ'
                return 'ɪ'
            if char == 'у':
                return 'ʊ'
            return base

        soft_vowels = {'е', 'ё', 'и', 'ю', 'я'}
        iotated = {'е': 'e', 'ё': 'o', 'ю': 'u', 'я': 'a'}

        phones: list[str] = []
        i = 0
        while i < len(word):
            char = word[i]
            prev_char = word[i - 1] if i > 0 else ''
            next_char = word[i + 1] if i + 1 < len(word) else ''

            # Мягкий/твёрдый знак — не порождают фонем
            if char in {'ь', 'ъ'}:
                i += 1
                continue

            # Аффриката «дж»
            if word.startswith('дж', i):
                phones.append('dʐ')
                i += 2
                continue

            # Удвоенные согласные → одна фонема
            if char == prev_char and char not in RUSSIAN_VOWELS and i > 0:
                i += 1
                continue

            # Йотированная гласная в начале слова или после гласной/ь/ъ
            if char in iotated and (i == 0 or prev_char in RUSSIAN_VOWELS
                                    or prev_char in {'ь', 'ъ'}):
                phones.extend(['j', vowel_phone(char, i, iotated[char], prev_char)])
                i += 1
                continue

            # Смягчение согласного перед гласными переднего ряда или ь
            if char in SOFT_CONSONANTS and (next_char in soft_vowels or next_char == 'ь'):
                phones.append(SOFT_CONSONANTS[char])
                i += 1
                continue

            # Базовая буква
            if char in LETTER_TO_PHONE:
                base = LETTER_TO_PHONE[char]
                if char in RUSSIAN_VOWELS:
                    phones.append(vowel_phone(char, i, base, prev_char))
                else:
                    phones.append(base)
            i += 1

        return self.devoice_consonants(phones)

    # ---- Основной метод ----

    def transcribe(self, tokens: list[str] | np.ndarray) -> list[list[str]]:
        """Transcribe each token of one sentence."""
        result: list[list[str]] = []
        for token in tokens:
            token = str(token)
            if token == SIL:
                result.append([SIL])
                continue

            key = self.normalize(token)

            # 1. Словарь RUSLAN
            if key in self.lexicon:
                best_phones = self.lexicon[key].most_common(1)[0][0]
                result.append(best_phones.split())
                continue

            # 2. Словарь MFA
            if key in self.mfa_dict:
                result.append(self.mfa_dict[key].split())
                continue

            # 3. Residual OOV — правила
            result.append(self.transcribe_oov(key))

        return result


# ---- Загрузка MFA-словаря ----

def load_mfa_dictionary(path: str) -> dict[str, str]:
    """Read a Russian MFA pronunciation dictionary (first variant per word)."""
    mfa_dict: dict[str, str] = {}
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            word = unicodedata.normalize('NFC', parts[0]).lower()
            phones = unicodedata.normalize('NFC', parts[-1])
            if word not in mfa_dict:
                mfa_dict[word] = phones
    return mfa_dict


# ---- Метрики ----

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


# ---- Загрузка данных ----

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


# ---- Оценка ----

def calc_metrics(g2p: G2P, tokens: pd.DataFrame, lexicon: dict[str, Counter]) -> None:
    """Print running-text PER, and PER and word accuracy on OOV tokens."""
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

    mfa_dict = load_mfa_dictionary(MFA_DICT_PATH)
    print(f'MFA dictionary: {len(mfa_dict)} entries')

    g2p = G2P(lexicon, mfa_dict)

    print('\nCalculate metrics, training fold:')
    calc_metrics(g2p, train, lexicon)

    print('\nCalculate metrics, testing fold:')
    calc_metrics(g2p, test, lexicon)

    residual = test[
        ~test.token.str.lower().isin(lexicon)
        & ~test.token.str.lower().isin(mfa_dict)
    ]
    residual_ref = [p.split() for p in residual.phones]
    residual_hyp = [g2p.transcribe([w])[0] for w in residual.token]
    print(f'\nResidual OOV: {len(residual)} tokens, '
          f'{residual.token.nunique()} unique words')
    print(f'Residual OOV PER {phone_error_rate(residual_ref, residual_hyp):.4f}')
    print(f'Residual OOV word accuracy {word_accuracy(residual_ref, residual_hyp):.4f}')


if __name__ == '__main__':
    test_g2p()