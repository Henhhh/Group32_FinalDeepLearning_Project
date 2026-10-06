# Visual Question Answering for Receipts and Documents

A system that answers English questions about document and receipt images. It compares Tesseract OCR + RoBERTa QA with Donut (Swin encoder + mBART decoder). The final results use the cleaned dataset and the `donut_decoder_finetuned_clean` checkpoint.

## Data and models

Download the two ZIP files linked in `DATA.md`. Extract the dataset to `data_clean/` and the final checkpoint to `checkpoints/donut_decoder_finetuned_clean/`. Model weights and dataset images are provided separately and are not included in the source package.

- Training: 167 questions from 24 documents.
- Validation: 43 questions from 12 documents.
- Evaluation: 57 questions from 12 documents.
- The three DocVQA splits have no overlapping document IDs within this project's additional fine-tuning data.
- Receipts: 20 receipts and 40 questions, used for evaluation only.

## Colab demo

Open `notebooks/Run_Project_Colab.ipynb`. Upload the updated source ZIP, named `Document_VQA_Project.zip`, to `/content/`. Run the first five code cells: extract the source and mount Drive, install dependencies, check the checkpoint, start Flask, and display the interface.

The notebook expects the extracted source folder to be named `Document_VQA_Project`. If you rename that folder, update `CODE_DIR` and the installation path accordingly. Place the extracted model in your own Drive at `MyDrive/Document_VQA_Project/checkpoints/donut_decoder_finetuned_clean/`, or update `PROJECT_DATA_DIR` to its actual location.

The model path must be:

```python
MODEL_DIR = PROJECT_DATA_DIR / 'checkpoints/donut_decoder_finetuned_clean'
```

A GPU speeds up inference; the demo also supports CPU through `device='auto'`. The demo has been checked on CPU. CPU demo timings do not replace the T4 evaluation timings. Retraining is not required to run the demo. The first request for each method loads its model; RoBERTa requires Internet access if it has not been downloaded. Demo screenshots are stored in `docs/demo/`.

## Environment

The final evaluation ran on Colab with a Tesla T4: Python 3.13, PyTorch 2.11.0+cu130, Transformers 5.18.0, and Hugging Face Hub 1.33.0. `results/environment.json` records the saved environment information. `requirements.txt` specifies Transformers 5.18.0 and Hugging Face Hub 1.33.0 and does not include the former tokenizer 0.21 constraint. The full dependency environment is not locked.

## Downloading assets and running the demo on Windows

### 1. Download the source code

On GitHub, select **Code → Download ZIP**, extract the archive, and open the project folder in VS Code.

Open a terminal in the folder containing `app.py` and `requirements.txt`.

### 2. Download the dataset and checkpoint

Open `DATA.md` and download:

- `processed_dataset_clean.zip`
- `donut_decoder_finetuned_clean.zip`

Extract the archives and place their contents as follows:

```text
Document_VQA_Project/
├── app.py
├── requirements.txt
├── data_clean/
│   ├── train.json
│   ├── validation.json
│   ├── evaluation.json
│   ├── images/
│   └── receipts/
└── checkpoints/
    └── donut_decoder_finetuned_clean/
        ├── config.json
        ├── ... model weights ...
        └── ... processor and tokenizer files ...
```

If an archive contains an extra wrapper folder, move the inner folder to the location shown above. Avoid nesting two folders named `data_clean` or `donut_decoder_finetuned_clean`.

The dataset is used for evaluation and provides sample images. If you only want to demonstrate the system with your own images, download the checkpoint.

### 3. Install Python dependencies

Run these commands in the VS Code terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

To use OCR + RoBERTa QA, also install the Tesseract OCR program on Windows with English (`eng`) language data. Installing `pytesseract` through pip does not install the Tesseract program.

### 4. Start the demo

```powershell
.\.venv\Scripts\python.exe app.py --model-path checkpoints/donut_decoder_finetuned_clean --device auto
```

If Tesseract is not on PATH, specify its actual installation path, for example:

```powershell
.\.venv\Scripts\python.exe app.py --model-path checkpoints/donut_decoder_finetuned_clean --device auto --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

`auto` selects a CUDA GPU when available and otherwise uses CPU. CPU inference may take longer.

### 5. Try question answering

Open http://127.0.0.1:5000 in a browser.

- Select a document or receipt image.
- Select Donut or OCR + RoBERTa QA.
- Enter an English question about information visible in the image.
- Submit the question and inspect the answer.

Sample images are available in `data_clean/images/` and `data_clean/receipts/images/`. Questions and reference answers are stored in the corresponding `evaluation.json` files.

Images in `docs/demo/` are result screenshots, not input images for testing. They preserve the original Vietnamese interface as historical evidence; the interface in this English source package has been translated.

The first request for each method requires model loading. The RoBERTa baseline requires Internet access to download its model if it is not cached. Keep the terminal running while using the demo.

The demo has been checked on Colab CPU. The Windows instructions still need to be verified in a fresh local environment.

## Final DocVQA results

| Method | Correct | EM (%) | ANLS | Seconds/question |
|---|---:|---:|---:|---:|
| OCR + RoBERTa QA | 13/57 | 22.81 | 0.2628 | 3.15 |
| Donut before additional fine-tuning | 40/57 | 70.18 | 0.7739 | 1.53 |
| Donut after fine-tuning on clean data | 42/57 | 73.68 | 0.7948 | 1.49 |

The main table is `results/all_metrics.csv`. Final raw JSON files and summary tables are stored in `results/`. Historical results are stored in `results/legacy/`. See `docs/REPORT_NOTES.md` for interpretation.

## Re-running evaluation and result summaries

Run commands from the project root. Local paths may be replaced with absolute Drive paths.

```bash
python -m src.evaluate --method donut --model-path checkpoints/donut_decoder_finetuned_clean --data-file data_clean/evaluation.json --output results/donut_after_finetune_recheck.json
python -m src.evaluate --method ocr --data-file data_clean/evaluation.json --output results/ocr_qa_recheck.json
python -m src.summarize_results --results-dir results
```

The recheck files do not automatically replace the seven main input files used by `summarize_results`. For reduced-resolution evaluation, add `--transform downsample50`. For receipts, use `data_clean/receipts/evaluation.json`. Do not overwrite the final raw results. Timings include preprocessing and inference and exclude model loading, warm-up, image reading, and degradation-image creation. Do not run evaluations concurrently when measuring runtime.

## Retraining

Download the initial model, `DonutDocVQA.rar`, through `DATA.md` and extract it to `checkpoints/DonutDocVQA/`. Download the cleaned dataset and place it in `data_clean/`.

Run the following command from the project root in an environment with the dependencies installed and a CUDA GPU. Start from the initial model, not the project's final checkpoint.

```bash
python -m src.train_donut --data-dir data_clean --model-path checkpoints/DonutDocVQA --output-dir checkpoints/donut_decoder_retrained_clean --results-dir results/retrained_clean --epochs 2 --learning-rate 1e-5
```

Training uses a frozen encoder, decoder fine-tuning, batch size 1, gradient accumulation 4, FP16, seed 42, weight decay 0.01, and maximum sequence length 128. The checkpoint is selected by validation loss; evaluation is not used to select an epoch. Optimizer state is not saved for resuming between epochs. Different GPUs or libraries may produce numerical differences.

`src/prepare_data.py` creates the ordered pilot sample and detects document overlap, but does not automatically remove the 33 overlapping training questions. To reproduce the final filtering step from the original data, use `notebooks/Check_And_Clean_Dataset.ipynb`. The `data_clean` ZIP already contains the checked, cleaned data. Retraining and evaluation history are stored in `notebooks/Retrain_Clean_Colab.ipynb`.

## Main files

- `src/train_donut.py`: decoder fine-tuning.
- `src/donut_inference.py`: answer generation from an image and a question.
- `src/ocr_qa.py`: Tesseract + AutoTokenizer/AutoModelForQuestionAnswering; the former question-answering pipeline is not used.
- `src/evaluate.py`, `src/metrics.py`, `src/summarize_results.py`: evaluation and result summaries.
- `app.py`, `templates/index.html`: Flask demo.
- `DATA.md`: sources, splits, preprocessing, and download links.
- `docs/REPORT_NOTES.md`, `docs/VALIDATION.md`: results, limitations, and verification scope.

Sources: [DocVQA](https://huggingface.co/datasets/pixparse/docvqa-single-page-questions), [CORD](https://huggingface.co/datasets/naver-clova-ix/cord-v2), [Donut paper](https://arxiv.org/abs/2111.15664), and [RoBERTa QA](https://huggingface.co/deepset/roberta-base-squad2).
