import argparse
import json
import time
from pathlib import Path
from PIL import Image
from src.common import read_json, save_json, degrade_image
from src.metrics import score_row, summarize

def sync_gpu():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def run_evaluation(engine, rows, image_root, output, transform='original'):
    if not rows:
        raise ValueError('Empty evaluation set')
    if len({r['question_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate question IDs')
    if Path(output).exists():
        raise FileExistsError(f'Result exists: {output}. Choose a new filename to preserve experiments.')
    def load(row):
        with Image.open(Path(image_root) / row['image_path']) as img:
            image = img.convert('RGB')
        return degrade_image(image) if transform == 'downsample50' else image
    engine.predict(load(rows[0]), rows[0]['question'])
    sync_gpu()
    predictions = []
    for i, row in enumerate(rows, 1):
        image = load(row)
        sync_gpu()
        start = time.perf_counter()
        prediction = engine.predict(image, row['question'])
        sync_gpu()
        elapsed = time.perf_counter() - start
        scored = score_row({**row, **{k:v for k,v in prediction.items() if k != 'answer'},
                            'prediction': prediction['answer'], 'seconds': elapsed,
                            'transformation': transform})
        predictions.append(scored)
        save_json(output, predictions)
        print(f'{i}/{len(rows)}', flush=True)
    return summarize(predictions)

def main():
    parser = argparse.ArgumentParser(description='Evaluate a system on original or degraded document/receipt images')
    parser.add_argument('--method', choices=['donut','ocr'], required=True)
    parser.add_argument('--model-path', help='Donut checkpoint folder or Hugging Face model ID')
    parser.add_argument('--qa-model', default='deepset/roberta-base-squad2')
    parser.add_argument('--data-file', required=True)
    parser.add_argument('--image-root', help='Defaults to the parent folder of data-file')
    parser.add_argument('--output', required=True)
    parser.add_argument('--transform', choices=['original','downsample50'], default='original')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--tesseract-cmd')
    args = parser.parse_args()
    if args.method == 'donut':
        if not args.model_path:
            parser.error('--model-path is required for Donut')
        from src.donut_inference import DonutQA
        engine = DonutQA(args.model_path, args.device)
    else:
        from src.ocr_qa import OCRQA
        engine = OCRQA(args.qa_model, args.device, args.tesseract_cmd)
    result = run_evaluation(engine, read_json(args.data_file),
                            args.image_root or Path(args.data_file).parent,
                            args.output, args.transform)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
