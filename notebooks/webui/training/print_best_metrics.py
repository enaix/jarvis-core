"""Quick: print P/R/F1 для всех LogReg configs, mixed + gold."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lib import (
    load_split, assemble_features, binary_y, filter_labels,
    train_logreg, evaluate,
)
import numpy as np

train = load_split('train')
test = load_split('test')
keep_train = filter_labels(train, exclude_skip=True)
keep_test = filter_labels(test, exclude_skip=True)


def restrict(d, idx):
    out = {'labels': [d['labels'][i] for i in idx]}
    for key in ('geometric', 'structural', 'recurrence', 'role_onehot'):
        out[key] = d[key][idx]
    for key in ('word_tfidf', 'char_tfidf'):
        out[key] = d[key][idx]
    return out


tr = restrict(train, keep_train)
te = restrict(test, keep_test)

CONFIGS = [
    ('text only',           ['text']),
    ('text+geo',            ['text', 'geometric']),
    ('text+geo+struct',     ['text', 'geometric', 'structural', 'role']),
    ('text+geo+struct+rec', ['text', 'geometric', 'structural', 'role', 'recurrence']),
]

print(f"{'config':<22s} | {'mixed':<35s} | {'gold':<35s}")
print(f"{'-'*22} | {'-'*35} | {'-'*35}")
for name, groups in CONFIGS:
    X_train = assemble_features(tr, groups)
    y_train = binary_y(tr['labels'])
    X_test = assemble_features(te, groups)
    y_test = binary_y(te['labels'])

    model = train_logreg(X_train, y_train)
    mixed = evaluate(model, X_test, y_test, te['labels'], gold_only=False)
    gold = evaluate(model, X_test, y_test, te['labels'], gold_only=True)

    mstr = f"F1={mixed['f1']:.3f} P={mixed['precision']:.3f} R={mixed['recall']:.3f} n={mixed['n_samples']}"
    gstr = f"F1={gold['f1']:.3f} P={gold['precision']:.3f} R={gold['recall']:.3f} n={gold['n_samples']}"
    print(f"{name:<22s} | {mstr:<35s} | {gstr:<35s}")
