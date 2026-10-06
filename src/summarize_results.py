import argparse
import csv
from pathlib import Path
from src.common import read_json, save_json
from src.metrics import score_row, summarize

def main():
    parser = argparse.ArgumentParser(description='Score existing JSON predictions without running models')
    parser.add_argument('--results-dir', required=True)
    args = parser.parse_args()
    root = Path(args.results_dir)
    entries = []
    document_ids = None
    for name in ['ocr_qa_baseline','donut_before_finetune','donut_after_finetune',
                 'ocr_qa_downsample50','donut_after_finetune_downsample50',
                 'receipt_ocr_qa','receipt_donut_after_finetune']:
        path = root / f'{name}.json'
        if not path.exists():
            print('Missing:', path)
            continue
        rows = read_json(path)
        ids = {r['question_id'] for r in rows}
        if len(ids) != len(rows):
            raise ValueError(f'Duplicate IDs: {path}')
        if not name.startswith('receipt_'):
            if document_ids is None:
                document_ids = ids
            elif ids != document_ids:
                raise ValueError('Document result files use different question sets')
        save_json(root / f'{name}_scored.json', [score_row(r) for r in rows])
        entries.append({'experiment': name, **summarize(rows)})
        if name.startswith('receipt_'):
            for kind in sorted({r['question_type'] for r in rows}):
                entries.append({'experiment': name + ':' + kind,
                                **summarize([r for r in rows if r['question_type'] == kind])})
    if not entries:
        raise ValueError('No known prediction files found')
    with (root / 'all_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(entries[0]))
        writer.writeheader(); writer.writerows(entries)
    for r in entries:
        print(r)

if __name__ == '__main__':
    main()
