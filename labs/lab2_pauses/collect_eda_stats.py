"""Collect duration stats from all RUSLAN TextGrids for lab 2 EDA."""
import os
import pickle
import tqdm
from praatio import textgrid

ALIGN_DIR = "../../data/RUSLAN_align_v2"
OUT_PATH = "data/eda_stats.pkl"

VOWELS = set("aeiouyæɛɪʊʌəɨɐɑɒɔɜœøɘɵɤɯʉɶ")


def strip_diacritics(label: str) -> str:
    """Убираем комбинирующиеся диакритики (U+0300..U+036F)."""
    return "".join(c for c in label if not (0x0300 <= ord(c) <= 0x036F))


def main():
    files = sorted(f for f in os.listdir(ALIGN_DIR) if f.endswith(".TextGrid"))
    print(f"Files: {len(files)}")

    pauses = []           # паузы внутри фразы (между словами, включая пустые интервалы)
    words = []            # длительности слов
    final_pauses = []     # финальные паузы (пустой интервал в конце или разрыв до xmax)
    phones = []           # длительности всех фонем
    vowels = []           # длительности гласных фонем
    n_errors = 0

    for fname in tqdm.tqdm(files):
        try:
            tg = textgrid.openTextgrid(
                os.path.join(ALIGN_DIR, fname), includeEmptyIntervals=True
            )
        except Exception:
            n_errors += 1
            continue

        words_tier = tg.getTier("words")
        phones_tier = tg.getTier("phones")
        xmax = tg.maxTimestamp

        entries = words_tier.entries
        prev_end = None

        for i, e in enumerate(entries):
            is_last = (i == len(entries) - 1)

            if e.label == "":
                # Пустой интервал — пауза
                if is_last:
                    # Последний интервал пустой → финальная пауза
                    final_pauses.append(e.end - e.start)
                else:
                    if e.end - e.start > 0.001:
                        pauses.append(e.end - e.start)
                prev_end = e.end
                continue

            # Непустой — слово
            if prev_end is not None and e.start > prev_end + 0.001:
                # Разрыв между словами — тоже пауза
                pauses.append(e.start - prev_end)
            words.append(e.end - e.start)
            prev_end = e.end

            # Если это последний интервал в tier и есть разрыв до xmax — финальная пауза
            if is_last and xmax > e.end + 0.001:
                final_pauses.append(xmax - e.end)

        # Фонемы
        for p in phones_tier.entries:
            dur = p.end - p.start
            if dur <= 0 or p.label == "":
                continue
            phones.append(dur)
            base = strip_diacritics(p.label)[:1].lower()
            if base in VOWELS:
                vowels.append(dur)

    data = {
        "pauses": pauses,
        "words": words,
        "final_pauses": final_pauses,
        "phones": phones,
        "vowels": vowels,
    }
    with open(OUT_PATH, "wb") as f:
        pickle.dump(data, f)

    print(f"\nПауз между словами: {len(pauses)}")
    print(f"Финальных пауз:     {len(final_pauses)}")
    print(f"Слов:               {len(words)}")
    print(f"Фонем:              {len(phones)}")
    print(f"Гласных:            {len(vowels)}")
    print(f"Ошибок чтения:      {n_errors}")


if __name__ == "__main__":
    main()