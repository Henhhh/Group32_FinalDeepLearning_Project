import argparse

def main():
    parser = argparse.ArgumentParser(description='Create receipt QA from CORD test labels')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()
    import io
    import json
    from pathlib import Path
    from datasets import load_dataset
    from PIL import Image

    RECEIPT_DIR = Path(args.data_dir)
    (RECEIPT_DIR / "images").mkdir(parents=True, exist_ok=True)

    cord_test = load_dataset(
        "naver-clova-ix/cord-v2",
        split="test",
        streaming=True, revision=args.revision,
    )

    question_templates = [
        ("total", "total_price",
         "What is the total amount on the receipt?"),
        ("sub_total", "subtotal_price",
         "What is the subtotal amount on the receipt?"),
        ("sub_total", "tax_price",
         "What is the tax amount on the receipt?"),
    ]

    def to_pil(value):
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
            return Image.open(value["path"]).convert("RGB")
        if isinstance(value, bytes):
            return Image.open(io.BytesIO(value)).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(value)}")

    receipt_records = []

    for index, sample in enumerate(cord_test.take(args.count)):
        ground_truth = sample["ground_truth"]
        if isinstance(ground_truth, str):
            ground_truth = json.loads(ground_truth)

        parsed = ground_truth["gt_parse"]
        image = to_pil(sample["image"])
        image_path = f"images/receipt_{index:03d}.png"
        image.save(RECEIPT_DIR / image_path)

        for group, field, question in question_templates:
            fields = parsed.get(group, {})
            if not isinstance(fields, dict):
                continue

            answer = fields.get(field)
            if not isinstance(answer, (str, int, float)):
                continue
            if not str(answer).strip():
                continue

            receipt_records.append({
                "question_id": f"cord_test_{index:03d}_{field}",
                "document_id": f"cord_test_{index:03d}",
                "image_path": image_path,
                "question": question,
                "answers": [str(answer).strip()],
                "question_type": field,
                "source": "naver-clova-ix/cord-v2",
                "source_split": "test",
            })

    assert receipt_records, "Could not create questions from the annotations."

    (RECEIPT_DIR / "evaluation.json").write_text(
        json.dumps(receipt_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Number of receipts with questions:",
          len({r["document_id"] for r in receipt_records}))
    print("Number of questions:", len(receipt_records))


if __name__ == '__main__':
    main()
