# Driver Drowsiness Detection — EfficientNetB0 + FastKAN

**Student:** *[Gulomjon Odamov]*
**Live demo:** https://driver-drowsiness-detection-1997.streamlit.app
**Repository:** https://github.com/Odamov490/driver-drowsiness-detection

## Problem statement
Driver drowsiness is a major cause of road accidents. This project detects
drowsiness from a driver's face image and classifies it into four states, so a
driver-monitoring context can flag unsafe conditions early.

## Selected project track
Computer Vision — image classification with a trained deep-learning model.

## Dataset source
NTHU-DDD Multi-Class (Kaggle: `samymesbah/nthu-dataset-ddd-multi-class`),
loaded reproducibly via `kagglehub`. Total: 66,521 labeled face frames.

## ML task type
Multi-class image classification (4 classes):
`notdrowsy`, `sleepyCombination`, `slowBlinkWithNodding`, `yawning`.

## Project pipeline / system architecture
Raw image → resize to 224×224 + ImageNet normalization → EfficientNetB0 backbone
(ImageNet-pretrained) → FastKAN classification head → softmax → predicted class +
confidence. The same preprocessing is reused at training and inference.

## Models / approaches tested
- **EfficientNetB0 + FastKAN** (final model) — CNN backbone with a Kolmogorov–
  Arnold-style classification head.
- Class-weighted cross-entropy loss to handle class imbalance.
- (A plain EfficientNetB0 baseline and a subject-independent split are explored
  in a separate corrected notebook for comparison.)

## Final model and justification
EfficientNetB0 + FastKAN was chosen for a strong, lightweight pretrained backbone
combined with an expressive, low-parameter classification head. It converged
stably over 20 epochs and gave balanced per-class performance on the test split.

## Evaluation metrics and results
Evaluated on a held-out test split of 9,979 images:

| Metric | Value |
|---|---|
| Accuracy | 98.78% |
| Precision (weighted) | 98.79% |
| Recall (weighted) | 98.78% |
| F1 (weighted) | 98.78% |

Per-class F1 ranges from 0.979 (yawning) to 0.998 (sleepyCombination). A full
classification report and confusion matrix are produced in the notebook.

## Installation instructions
```bash
git clone https://github.com/Odamov490/driver-drowsiness-detection.git
cd driver-drowsiness-detection
git lfs install        # model weights are stored via Git LFS
pip install -r requirements.txt
```

## Training / fine-tuning instructions
Open `Driver_Drowsiness_Detection_last.ipynb` in Google Colab (GPU runtime) and
run the cells top to bottom. The dataset downloads automatically via `kagglehub`;
the best checkpoint is saved to `./models/best_EfficientNetB0_FastKAN.pth`.

## Demo and inference run instructions (Colab-first)
- **Colab / notebook:** run the notebook's evaluation and single-image cells.
- **Local app:** `python smoke_test.py` then `streamlit run app.py`.
- **Public app:** open the live demo link, upload a face image, press **Predict**.

## Example input and output
- **Input:** one face image (jpg/png).
- **Output:** predicted class (e.g. `yawning`) with a confidence score and a
  per-class probability bar chart.

## Known limitations
- The train/test split is per-frame; because frames come from videos, similar
  frames can appear in both sets, which can inflate the reported accuracy. A
  subject-independent (driver-based) split is the recommended next step.
- The model is tuned to NTHU-DDD near-infrared, close-up footage, so ordinary
  daytime phone/webcam photos are out-of-distribution and can be misclassified.

## Responsible AI considerations
This is an educational demo, not a certified safety system, and must not be used
for real safety-critical driving decisions. The distribution limitation and
intended-use caveat are shown to users directly in the app's "Known limitation"
section.

## Reproducibility
`requirements.txt` pins dependencies; `smoke_test.py` verifies the model loads
and runs; the notebook is Colab-runnable end to end without any private Google
Drive dependency.
