"""Генерация графиков и метрик для отчёта v4 (без seaborn)."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data' / 'RUSLAN_pause_metadata.csv'
PICS = HERE / 'pics'
PICS.mkdir(exist_ok=True)

print('Читаю:', DATA)
df = pd.read_csv(DATA, sep='|', quoting=csv.QUOTE_NONE)
print('Строк:', len(df))

train = df[df.set == 'train'].copy()
work = train[train.is_last_word == 0].copy()
pauses = work[work.is_pause_after == 1].copy()
print('train work:', len(work), '| pauses:', len(pauses))

# График 1: длительности пауз
plt.figure(figsize=(8, 4))
plt.hist(pauses.pause_duration * 1000, bins=60, color='#4C72B0', edgecolor='white')
mean_p = (pauses.pause_duration * 1000).mean()
med_p = (pauses.pause_duration * 1000).median()
plt.axvline(mean_p, color='red', ls='--', label=f'mean = {mean_p:.0f} мс')
plt.axvline(med_p, color='green', ls='--', label=f'median = {med_p:.0f} мс')
plt.xlabel('длительность паузы, мс')
plt.ylabel('количество')
plt.title('Распределение длительностей пауз (train)')
plt.legend()
plt.tight_layout()
plt.savefig(PICS / 'dur_pause.png', dpi=110)
plt.close()
print('OK: dur_pause.png')

# График 2: длительности слов
plt.figure(figsize=(8, 4))
plt.hist(train.duration * 1000, bins=80, color='#DD8452', edgecolor='white')
mean_w = (train.duration * 1000).mean()
plt.axvline(mean_w, color='red', ls='--', label=f'mean = {mean_w:.0f} мс')
plt.xlabel('длительность слова, мс')
plt.ylabel('количество')
plt.title('Распределение длительностей слов (train)')
plt.xlim(0, 2000)
plt.legend()
plt.tight_layout()
plt.savefig(PICS / 'dur_word.png', dpi=110)
plt.close()
print('OK: dur_word.png')

# График 3: доля паузы по пунктуации
TRAILING = re.compile(r'([.,!?;:…—–\-»«")]+)$')
PUNCT2CLS = {
    '.': 'period', ',': 'comma', '?': 'question', '!': 'exclaim',
    '—': 'dash', '–': 'dash', '-': 'dash', '…': 'ellipsis',
    ';': 'semicolon', ':': 'colon', '»': 'closing', '«': 'closing',
    ')': 'closing', '"': 'other',
}
def cls(t):
    m = TRAILING.search(str(t))
    if not m: return 'none'
    for ch in m.group(1):
        if ch in PUNCT2CLS: return PUNCT2CLS[ch]
    return 'other'
work = work.copy()
work['punct_class'] = work.label_raw.apply(cls)
by = work.groupby('punct_class').agg(
    n=('label', 'size'),
    rate=('is_pause_after', 'mean'),
).sort_values('rate', ascending=False)

plt.figure(figsize=(9, 4.5))
plt.bar(by.index, by.rate, color='#4C72B0')
for i, (idx, row) in enumerate(by.iterrows()):
    plt.text(i, row.rate + 0.01, f'{row.rate:.2f}\n(n={row.n})', ha='center', fontsize=8)
plt.ylim(0, 1.1)
plt.ylabel('P(пауза после токена)')
plt.title('Доля пауз по классам пунктуации (train)')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(PICS / 'cls_vs_no.png', dpi=110)
plt.close()
print('OK: cls_vs_no.png')

# Метрики v4
metrics = {
    'train': {'F1': 0.7862, 'PRC': 0.7999, 'REC': 0.7729, 'MAE': 0.0908},
    'test':  {'F1': 0.7146, 'PRC': 0.7319, 'REC': 0.6981, 'MAE': 0.1008},
    'threshold': 0.51,
}
with open(HERE / 'data' / 'metrics_v4.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print('OK: metrics_v4.json')
print('\nГотово.')
