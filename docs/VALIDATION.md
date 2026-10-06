# Verification scope

- Python syntax for source version 7 was checked in the packaging environment.
- Unit/mock checks were recorded during an earlier packaging run; a full rerun on Transformers 5.18.0 is not claimed. Mock tests do not produce actual model metrics.
- The project author checked the data on Drive: required fields and referenced images are present, question IDs are unique within each split, and the three `data_clean` splits have no overlapping document IDs.
- The project author retrained from the initial DonutDocVQA model using 167 training questions, 43 validation questions, and 2 epochs. The process returned code 0, and the checkpoint contains configuration and weights.
- Seven experiments were run in the final Colab T4 environment. Raw JSON files and `all_metrics.csv` are stored in `results/`.
- The project author ran the CPU demo with the clean checkpoint. Screenshots are stored in `docs/demo/`; CPU timings are not used as substitutes for T4 timings.
- Model weights and dataset images are provided in separate Drive ZIP files and are not included in this source package. Their contents and download permissions must be checked against the links in `DATA.md`.
- `notebooks/Check_And_Clean_Dataset.ipynb` includes actual verification output: 33 training questions associated with overlapping documents were removed, leaving 167 training, 43 validation, and 57 evaluation questions. The three splits have no overlapping document IDs. The existing `data_clean` matches the filtering result and contains all referenced images.
- `notebooks/Retrain_Clean_Colab.ipynb` preserves retraining and evaluation history on clean data.
- `notebooks/Document_VQA_Project.ipynb` is retained as the experiment history on the original split.
- The source dataset commit was not recorded, and the full dependency environment is not locked. The demo was checked on Colab, but execution on a personal computer has not been verified.
- The English package translates documentation, comments, interface labels, and notebook messages. Saved notebook messages are translations of recorded output, not evidence of another model run. Original screenshots and numerical result files are preserved.
- Translation checks confirmed that Python and notebook code structure and numeric constants were preserved, and result files and screenshots remained byte-for-byte identical. Six unit tests passed in the translation environment; four Flask tests were skipped because Flask was not installed. Model inference and training were not rerun.
