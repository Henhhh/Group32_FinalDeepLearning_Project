# Dataset, checkpoints, and preprocessing

## Dataset and model downloads

- Dataset: [processed_dataset_clean.zip](https://drive.google.com/file/d/1f4xjKfBCfWqY7S674hPGjC9XfSgpMrnz/view?usp=sharing)
- Final checkpoint: [donut_decoder_finetuned_clean.zip](https://drive.google.com/file/d/1PSXyQHcmYW2htpAR5fBdLuxBPIxAlMi5/view?usp=sharing)
- Initial model: [DonutDocVQA.rar](https://drive.google.com/file/d/1GWaE8Fp5LXa37XaE9YGG9kLZXnoOtGUz/view?usp=sharing), used for retraining and evaluating Donut before additional fine-tuning.

Extract the dataset to `data_clean/`, containing `train.json`, `validation.json`, `evaluation.json`, `images/`, and `receipts/`. Extract the final checkpoint to `checkpoints/donut_decoder_finetuned_clean/`, containing the configuration, model weights, processor, and tokenizer. Both ZIP files must allow downloads as required for submission. Do not replace these links with older ZIP links unless their contents have been updated.

Extract `DonutDocVQA.rar` to `checkpoints/DonutDocVQA/`. This folder must directly contain `config.json`, model weights, processor, and tokenizer files. Avoid an additional nested folder with the same name.

The initial model was already fine-tuned on DocVQA before this project. The `donut_decoder_finetuned_clean` checkpoint results from additional fine-tuning on this project's 167 cleaned training questions.

## DocVQA

Source: https://huggingface.co/datasets/pixparse/docvqa-single-page-questions . The original pilot used the first 200 questions from the source training split and the first 100 questions from the source validation split through streaming. This is ordered sampling, not random sampling, and is not representative of the full dataset. The source commit was not recorded; retain the processed-data ZIP to reproduce the exact sample.

Document IDs are taken from `other_metadata.ucsf_document_id`, then `doc_id`, then `image`. For the 100 source validation questions, SHA256(document_id) modulo 2 assigns local validation (0) and evaluation (1), producing 43 and 57 questions.

A subsequent check found three document IDs shared between training and the other splits: `ffbg0227`, `ffhx0227`, and `ffjw0228`. The 33 questions associated with these documents were removed from training. Validation and evaluation were preserved. The cleaned data were saved in `data_clean`, while the original data were retained.

| Final split | Questions | Documents |
|---|---:|---:|
| train | 167 | 24 |
| validation | 43 | 12 |
| evaluation | 57 | 12 |

The three splits have no overlapping document IDs. This controls overlap within this project's additional fine-tuning data; it has not been verified whether the evaluation documents appeared in the initial Donut checkpoint's prior training.

Images are RGB PNG files, with one image file per question, so several files may contain the same document image. `data_clean` may retain images that became unreferenced after training-data filtering; `train.json` determines the samples used for training. Data checks confirmed that referenced images exist, required fields are present and valid, and question IDs are not duplicated within each split.

Schema: `document_id`, `question_id`, `image_path` (relative to `data_clean/`), `question`, and `answers`. The model processor preserves the checkpoint's preprocessing. Training uses `answers[0]` and maximum sequence length 128. The final configuration records `train_used=167` and `validation_used=43`. Evaluation uses the best score across accepted reference answers.

Reproduction: use the checked `data_clean` ZIP for training and evaluation. Use `notebooks/Check_And_Clean_Dataset.ipynb` to reproduce filtering from the original data. `src.prepare_data` creates the pilot and checks overlap, but does not automatically filter the training split.

## Receipts

Source: https://huggingface.co/datasets/naver-clova-ix/cord-v2 . The dataset repository specifies CC BY 4.0; cite the authors and dataset. The first 20 test images were selected and converted into 40 English questions using annotation fields: 13 `subtotal_price`, 8 `tax_price`, and 19 `total_price`. Receipts are not used for training.

Nonempty scalar labels are converted into questions using fixed templates; missing fields are skipped. Currency punctuation and digits are preserved without normalizing monetary values. This is a custom QA conversion, not an official CORD QA benchmark. Images are stored in `data_clean/receipts/images/` and questions in `receipts/evaluation.json`. The baseline runs OCR on the actual images; ground truth is used only for evaluation.

## Models

Initial Donut model: `DonutDocVQA`, supplied by the project author, a VisionEncoderDecoderModel with a donut-swin encoder and mBART decoder. It was already fine-tuned on DocVQA before this project. Reference model card: https://huggingface.co/naver-clova-ix/donut-base-finetuned-docvqa . The local weight hash has not been compared with the Hub version.

The decoder was additionally fine-tuned on 167 questions with the encoder frozen, for 2 epochs, with LR 1e-5 and seed 42. The final checkpoint is `donut_decoder_finetuned_clean`. The baseline uses English Tesseract OCR + `deepset/roberta-base-squad2` through the tokenizer and QA model directly in Transformers 5.18.0. The new postprocessing may differ from the previous pipeline.

## Robustness and metrics

`downsample50` reduces each image dimension to floor(size/2) using LANCZOS, then restores the original size using BICUBIC. Both methods receive the same transformation; labels and original images are preserved.

EM: lowercase, trim, and collapse whitespace; preserve punctuation and digits. ANLS: lowercase and trim; normalize Levenshtein distance by the maximum string length. The score is 1-distance if distance<0.5, otherwise 0. Take the maximum score over references and average over questions. ANLS ranges from 0 to 1. A one-digit error may still receive ANLS credit, so interpret it together with EM.

Timing includes processor + generation or OCR + QA. It excludes model loading, warm-up, image reading, and degradation-image creation. Final results were measured on T4; CPU screenshots only demonstrate that the demo runs.
