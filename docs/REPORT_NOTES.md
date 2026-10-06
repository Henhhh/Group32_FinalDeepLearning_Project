# Report notes — final results on clean data

DocVQA: training 167 questions/24 documents, validation 43/12, and evaluation 57/12. The three splits have no overlapping document IDs. After removing 33 training questions from three overlapping documents, the model was retrained from the initial checkpoint. The receipt set contains 20 images/40 questions and is not used for training.

Evaluation environment: Colab T4, Transformers 5.18.0, PyTorch 2.11.0+cu130, and Hub 1.33.0. The baseline uses Tesseract OCR and the RoBERTa QA model directly, replacing the former pipeline.

| Experiment | Correct | EM (%) | ANLS | Seconds/question |
|---|---:|---:|---:|---:|
| ocr_qa_baseline | 13/57 | 22.81 | 0.2628 | 3.15 |
| donut_before_finetune | 40/57 | 70.18 | 0.7739 | 1.53 |
| donut_after_finetune | 42/57 | 73.68 | 0.7948 | 1.49 |

The decoder was additionally fine-tuned for 2 epochs with a frozen encoder, LR 1e-5, batch size 1, gradient accumulation 4, seed 42, and FP16. Epoch 0 is the initial evaluation, not a training epoch. Validation loss: 5.6370 → 2.9424 → 2.2973; epoch 2 produced the best checkpoint.

Donut answered two more questions correctly (40→42), improving EM by 3.51 percentage points and ANLS by approximately 0.0209. Lower loss does not guarantee a corresponding increase in EM; accuracy alone does not establish underfitting. The evaluation sample contains only 57 questions from 12 documents and is insufficient to establish general improvement or statistical significance.

| Experiment | Correct | EM (%) | ANLS | Seconds/question |
|---|---:|---:|---:|---:|
| ocr_qa_downsample50 | 14/57 | 24.56 | 0.2869 | 3.04 |
| donut_after_finetune_downsample50 | 41/57 | 71.93 | 0.7833 | 1.45 |
| receipt_ocr_qa | 2/40 | 5.00 | 0.0972 | 1.04 |
| receipt_donut_after_finetune | 27/40 | 67.50 | 0.7495 | 1.58 |

With downsample50, Donut lost one correct answer and OCR gained one. This does not establish that reducing resolution always benefits OCR. Small timing differences do not prove a speed improvement.

Receipt results for Donut: subtotal 12/13, tax 3/8, and total 12/19. OCR: subtotal 0/13, tax 0/8, and total 2/19. There are only 20 receipts with template-generated questions, so conclusions are limited to this test set. Inspect OCR text and predictions before attributing failures to OCR or QA.

ANLS measures string similarity; a monetary value with one incorrect digit can still receive credit. Therefore, report EM together with ANLS. Reported timings are from T4 and exclude model loading, warm-up, image reading, and degradation-image creation. CPU demo timings are separate.

The initial checkpoint was already fine-tuned on DocVQA. The cleaned split only guarantees no overlap within this project's fine-tuning data. It has not been verified that the model's prior training excluded the evaluation documents. Do not describe the initial model as never having learned from DocVQA.

Results from train200 and the previous environment are historical. Use `results/all_metrics.csv` and the new raw JSON files as the final evidence. The two ZIP links are in `DATA.md`; cleaning and retraining notebooks are in `notebooks/`; CPU demo screenshots are in `docs/demo/`.
