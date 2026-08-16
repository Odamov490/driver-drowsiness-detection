# Capstone Evidence Matrix — Driver Drowsiness Detection

> Rule: `Exact evidence location` must be a repository path + section/cell/output, a public link, or an exact live-demo step.

Project: **Driver Drowsiness Detection (EfficientNetB0 + FastKAN)**
Student: *[Gulomjon Odamov]*
Repository: `https://github.com/Odamov490/driver-drowsiness-detection`
Live demo: `https://driver-drowsiness-detection-1997.streamlit.app`

| Criterion | What I claim | Exact evidence location | Why this is proof | Status | Gap / risk | Next action |
|---|---|---|---|---|---|---|
| 1. Problem Definition and Project Alignment — 10 / min 6 | The project detects driver drowsiness from a face image and classifies it into four driver states (notdrowsy, sleepyCombination, slowBlinkWithNodding, yawning), aimed at driver-monitoring / road-safety use. | `README.md` → "Problem statement" and "ML task type" sections; notebook `Driver_Drowsiness_Detection_last.ipynb` Cell 2 (markdown) defining the 4-class task. | The problem, the user (driver-safety monitoring), and the ML task (image classification) are stated explicitly and match the implemented pipeline. | GREEN | — | — |
| 2. Data and Preprocessing Pipeline — 15 / min 9 | Uses the public NTHU-DDD Multi-Class dataset (66,521 images), inspects class distribution, and applies a resize→normalize preprocessing pipeline reused identically at inference. | Notebook Cell 6 output ("Total images: 66521" + per-class counts); Cells 8–10 (split + transforms); `src/inference.py` `EVAL_TRANSFORM`. | Data is loaded reproducibly via `kagglehub`, inspected (counts printed), and the same eval transform is used in training and in the deployed app. | GREEN | Random per-frame split can let similar frames share train/test (documented as a limitation). | Note the split limitation in README "Known limitations". |
| 3. Modeling and Experiments — 20 / min 12 | A real model (EfficientNetB0 backbone + FastKAN classification head) is trained for 20 epochs with class-weighted loss, not just an API call. | Notebook Cell 11–13 (model definition + training loop + class weights); training log in Cell 13 output (per-epoch loss/acc). | The architecture is defined and trained in-repo; class weights address imbalance; training curves show real convergence. | GREEN | Single architecture reported here; a plain-baseline comparison exists in a separate corrected notebook. | Optionally cite the baseline comparison notebook. |
| 4. Evaluation and Error Analysis — 15 / min 9 | Evaluated on a held-out test split (9,979 images) with accuracy, precision, recall, F1, a confusion matrix, and a per-class report. | Notebook Cell 15 output ("Accuracy : 98.78%" + classification report + confusion matrix figure). | Task-appropriate multi-class metrics are computed on data not used for training, with per-class breakdown. | GREEN | Weakest class in the report is `yawning` (precision 0.969) — visible in the per-class table. | Mention the weakest class during defense. |
| 5. End-to-End Implementation and Project Delivery — 20 / min 12 | A working end-to-end demo runs from an uploaded image to a predicted class + confidence, deployed publicly on Streamlit Cloud. | Live app `https://driver-drowsiness-detection-1997.streamlit.app`; `app.py` (UI) + `src/inference.py` (`predict_image`). | Anyone can upload an image and get a prediction from the trained weights in `artifacts/`; no local setup required. | GREEN | Cold start reloads the model (first request slower). | — |
| 6. Documentation and Reproducibility — 10 / min 6 | The repository documents setup, run, and Colab reproduction; the model is loaded from committed weights via Git LFS. | `README.md` (setup + Colab-first run steps); `requirements.txt`; `artifacts/best_EfficientNetB0_FastKAN.pth` (Git LFS); `smoke_test.py`. | Another person can install deps, run the smoke test, and launch the app from the repo alone. | GREEN | Model file is large (120 MB) → requires Git LFS. | README documents the Git LFS step. |
| 7. Responsible AI and Limitations — 5 / min 3 | The app states that the model is trained on NTHU-DDD (near-infrared, close-up) footage and may be unreliable on ordinary phone/webcam photos; it is not for real safety-critical use. | In-app "Known limitation" expander (`app.py`); README "Known limitations" and "Responsible AI considerations". | The domain/distribution limitation and intended-use caveat are shown to users directly in the product. | GREEN | — | — |
| 8. Presentation, Demo, and Q&A — 5 / min 3 | A 5-minute defense route and Q&A answers are prepared, with a live demo path from image upload to output. | `docs/defense_pitch_outline.md`; `docs/defense_question_bank.md`; live app. | The pitch and question bank map every claim to a locatable piece of evidence. | GREEN | — | Rehearse once before defense. |

## Essential requirements

| Requirement | Status | Exact evidence / blocker |
|---|---|---|
| At least 60/100 overall (official final scoring) | GREEN | All eight criteria have locatable GREEN evidence. |
| Trained or fine-tuned ML model | GREEN | `artifacts/best_EfficientNetB0_FastKAN.pth`; training loop in notebook Cell 13. |
| Evaluation on unseen data with task-appropriate metrics | GREEN | Notebook Cell 15: test-split accuracy/precision/recall/F1 + confusion matrix. |
| Working end-to-end demo | GREEN | Live Streamlit app + `app.py` / `src/inference.py`. |
| Clear reproduction instructions | GREEN | `README.md` setup/run + `requirements.txt` + `smoke_test.py`. |
| Final defense attendance + project Q&A | YELLOW | Attend defense; `docs/defense_question_bank.md` prepared. |

## Solo Show-Me-Where results

Blind locator rule: close the evidence, start from repo root/showcase, follow only the written location.

| # | High-risk claim | Written exact location | PASS/FAIL | Missing context | Rewrite / next action |
|---|---|---|---|---|---|
| 1 | Model reaches 98.78% test accuracy | Notebook `Driver_Drowsiness_Detection_last.ipynb` → Cell 15 output block | PASS | — | — |
| 2 | End-to-end demo works publicly | `https://driver-drowsiness-detection-1997.streamlit.app` → upload image → Predict | PASS | — | — |
| 3 | Limitation shown to users | `app.py` → "Known limitation" expander (rendered at bottom of the app) | PASS | — | — |
