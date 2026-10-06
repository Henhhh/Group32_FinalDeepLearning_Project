import argparse

def main():
    parser = argparse.ArgumentParser(description='Prepare the notebook DocVQA pilot split')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--train-count', type=int, default=200)
    parser.add_argument('--validation-count', type=int, default=100)
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()
    from datasets import load_dataset
    from pathlib import Path
    from PIL import Image
    from io import BytesIO
    import hashlib
    import json

    DATA_DIR = Path(args.data_dir)
    IMAGE_DIR = DATA_DIR / "images"
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    records = {"train": [], "validation": [], "evaluation": []}

    def to_image(value):
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, bytes):
            return Image.open(BytesIO(value))
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"]))
        return Image.open(value["path"])

    for source_split, limit in [("train", args.train_count), ("validation", args.validation_count)]:
        stream = load_dataset(
            "pixparse/docvqa-single-page-questions",
            split=source_split,
            streaming=True, revision=args.revision
        )

        for row in stream.take(limit):
            # Use document IDs so that questions from the same document
            # are always assigned to the same split.
            metadata = row["other_metadata"]
            document_id = str(
                metadata.get("ucsf_document_id")
                or metadata.get("doc_id")
                or metadata.get("image")
            )
            if document_id == "None":
                raise ValueError("Document ID not found")

            if source_split == "train":
                target_split = "train"
            else:
                group = int(
                    hashlib.sha256(document_id.encode()).hexdigest(), 16
                ) % 2
                target_split = "validation" if group == 0 else "evaluation"

            image_name = f"{source_split}_{row['question_id']}.png"
            to_image(row["image"]).convert("RGB").save(IMAGE_DIR / image_name)

            records[target_split].append({
                "document_id": document_id,
                "question_id": row["question_id"],
                "image_path": f"images/{image_name}",
                "question": row["question"],
                "answers": row["answers"]
            })

        print(f"Finished loading {source_split}")

    for split_name, rows in records.items():
        with open(DATA_DIR / f"{split_name}.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        print(f"{split_name}: {len(rows)} questions")

    assert records["validation"], "The validation split is empty"
    assert records["evaluation"], "The evaluation split is empty"

    validation_ids = {r["document_id"] for r in records["validation"]}
    evaluation_ids = {r["document_id"] for r in records["evaluation"]}

    assert validation_ids.isdisjoint(evaluation_ids)
    print("Validation and evaluation have no overlapping document IDs.")
    train_ids = {r['document_id'] for r in records['train']}
    if not train_ids.isdisjoint(validation_ids | evaluation_ids):
        raise ValueError('Source train overlaps validation/evaluation by document ID; inspect before training.')

if __name__ == '__main__':
    main()
