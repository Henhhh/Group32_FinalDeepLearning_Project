import torch
import pytesseract
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

class OCRQA:
    def __init__(self, model_id="deepset/roberta-base-squad2",
                 device="auto", tesseract_cmd=None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        selected = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else device

        self.device = torch.device(selected)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True
        )
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_id, use_safetensors=True
        ).to(self.device).eval()

    def predict(self, image, question):
        context = pytesseract.image_to_string(
            image.convert("RGB"), lang="eng",
            config="--oem 3 --psm 3"
        ).strip()

        if not context:
            return {"answer": "", "qa_score": 0.0, "ocr_text": ""}

        # Limit question length and split long text into windows
        q_ids = self.tokenizer(
            question, add_special_tokens=False
        )["input_ids"]
        if len(q_ids) > 64:
            question = self.tokenizer.decode(
                q_ids[:64], skip_special_tokens=True
            )

        encoded = self.tokenizer(
            question, context,
            max_length=384,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )

        offsets = encoded.pop("offset_mapping")
        encoded.pop("overflow_to_sample_mapping")
        best_score, best_answer = -1.0, ""

        with torch.inference_mode():
            for window in range(encoded["input_ids"].shape[0]):
                inputs = {
                    key: value[window:window + 1].to(self.device)
                    for key, value in encoded.items()
                }
                output = self.model(**inputs)

                sequence_ids = encoded.sequence_ids(window)
                context_mask = torch.tensor(
                    [value == 1 for value in sequence_ids],
                    device=self.device,
                    dtype=torch.bool,
                )
                valid = context_mask & inputs["attention_mask"][0].bool()

                # Allow the CLS token when normalizing probabilities,
                # but do not select an empty answer.
                cls_matches = (
                    inputs["input_ids"][0]
                    == self.tokenizer.cls_token_id
                ).nonzero(as_tuple=True)[0]
                allowed = valid.clone()
                if len(cls_matches):
                    allowed[cls_matches[0]] = True

                start = output.start_logits[0].masked_fill(
                    ~allowed, -10000.0
                ).softmax(dim=-1)
                end = output.end_logits[0].masked_fill(
                    ~allowed, -10000.0
                ).softmax(dim=-1)

                start = start.masked_fill(~valid, 0)
                end = end.masked_fill(~valid, 0)

                scores = start[:, None] * end[None, :]
                length = scores.shape[0]
                positions = torch.arange(
                    length, device=self.device
                )
                span_length = (
                    positions[None, :] - positions[:, None] + 1
                )
                scores = scores.masked_fill(
                    (span_length < 1) | (span_length > 30), 0
                )

                score, index = scores.flatten().max(dim=0)
                if score.item() > best_score and score.item() > 0:
                    first = index.item() // length
                    last = index.item() % length
                    begin = offsets[window, first, 0].item()
                    finish = offsets[window, last, 1].item()
                    best_answer = context[begin:finish]
                    best_score = score.item()

        return {
            "answer": best_answer.strip(),
            "qa_score": max(best_score, 0.0),
            "ocr_text": context,
        }
