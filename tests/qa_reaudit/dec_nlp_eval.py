"""Scores fundamentals.catalysts.extract_disclosures against the frozen gold corpus (tests/qa_reaudit/dec_corpus.py).

Matching rule: an extracted annotation is a true positive iff a gold span with the SAME tag overlaps it by at least
one character. Unmatched extractions are false positives, unmatched gold spans are false negatives.
"""
from __future__ import annotations
import importlib.util
import json
import pathlib
from collections import defaultdict

from fundamentals.catalysts import extract_disclosures, TAGS

HERE = pathlib.Path(__file__).parent


def load_corpus():
    spec = importlib.util.spec_from_file_location('dec_corpus_frozen', HERE / 'dec_corpus.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.corpus()


def overlaps(a, b):
    return a['start'] < b['end'] and b['start'] < a['end']


def evaluate(corpus=None):
    corpus = corpus or load_corpus()
    per_tag = {t: dict(tp=0, fp=0, fn=0) for t in sorted(TAGS)}
    rows = []
    for item in corpus:
        got = extract_disclosures(item['text'])
        gold = item['gold']
        matched_gold = set()
        item_fp, item_tp = [], []
        for a in got:
            hit = None
            for gi, g in enumerate(gold):
                if g['tag'] == a['tag'] and overlaps(a, g):
                    hit = gi
                    break
            if hit is None:
                item_fp.append(a)
                per_tag[a['tag']]['fp'] += 1
            else:
                matched_gold.add(hit)
                item_tp.append(a)
                per_tag[a['tag']]['tp'] += 1
        # a gold span can be matched by several extractions; count each gold once
        tp_gold = len(matched_gold)
        fn = [g for gi, g in enumerate(gold) if gi not in matched_gold]
        for g in fn:
            per_tag[g['tag']]['fn'] += 1
        rows.append(dict(id=item['id'], cat=item['cat'], dtype=item['dtype'], text=item['text'], gold=gold,
                         tp=item_tp, fp=item_fp, fn=fn))
    return per_tag, rows


def prf(d):
    tp, fp, fn = d['tp'], d['fp'], d['fn']
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    return p, r


def report(per_tag, rows):
    lines = []
    tot = dict(tp=0, fp=0, fn=0)
    for t, d in per_tag.items():
        p, r = prf(d)
        lines.append(f'{t:18s} tp={d["tp"]:3d} fp={d["fp"]:3d} fn={d["fn"]:3d} precision={p if p is None else round(p, 3)} recall={r if r is None else round(r, 3)}')
        for k in tot:
            tot[k] += d[k]
    p, r = prf(tot)
    lines.append(f'{"MICRO":18s} tp={tot["tp"]:3d} fp={tot["fp"]:3d} fn={tot["fn"]:3d} precision={p if p is None else round(p, 3)} recall={r if r is None else round(r, 3)}')
    lines.append('--- false positives')
    for r_ in rows:
        for a in r_['fp']:
            lines.append(f'FP {r_["id"]:4s} [{r_["cat"]}] {a["tag"]} quote={a["quote"]!r} dtype={r_["dtype"]} text={r_["text"]!r}')
    lines.append('--- false negatives')
    for r_ in rows:
        for g in r_['fn']:
            lines.append(f'FN {r_["id"]:4s} [{r_["cat"]}] {g["tag"]} text={r_["text"]!r}')
    return '\n'.join(lines), tot


if __name__ == '__main__':
    per_tag, rows = evaluate()
    text, tot = report(per_tag, rows)
    print(text)
    out = pathlib.Path('results/qa_reaudit_20261001/dec')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'nlp_scores.txt').write_text(text)
    (out / 'nlp_scores.json').write_text(json.dumps(dict(per_tag=per_tag, rows=rows), indent=1, default=str))
    import csv
    with open(out / 'nlp_per_item.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['id', 'category', 'doc_type', 'gold_tags', 'extracted_tp', 'extracted_fp', 'missed_fn', 'text'])
        for r in rows:
            w.writerow([r['id'], r['cat'], r['dtype'], ';'.join(g['tag'] for g in r['gold']), ';'.join(a['tag'] for a in r['tp']),
                        ';'.join(a['tag'] for a in r['fp']), ';'.join(g['tag'] for g in r['fn']), repr(r['text'])])
