# Driver Drowsiness Detection — Streamlit Deployment

A Streamlit application for the route:

`local project -> GitHub repository -> Streamlit Community Cloud -> public URL`

The model (EfficientNetB0 + FastKAN, trained on the NTHU-DDD dataset) is
already trained in a Colab notebook. This app loads the saved PyTorch
weights and runs inference on one uploaded image.

## Repository structure

```text
.
├── app.py                                    # Streamlit entrypoint
├── artifacts/
│   └── best_EfficientNetB0_FastKAN.pth       # Saved model weights (you add this)
├── src/
│   ├── __init__.py
│   └── inference.py                          # Model definition + validation + prediction
├── scripts/
│   └── (optional) retraining scripts
├── tests/
│   └── test_inference.py                     # Inference contract tests
├── .streamlit/config.toml                    # Visual configuration
├── requirements.txt                          # Cloud dependencies (CPU-only torch)
├── smoke_test.py                             # Fast pre-deployment check
└── PROJECT_STATUS.md                         # Route, fallback, and gate evidence
```

## 0. Add your model weights (do this first)

This repository ships without the trained weights (they are too large and
are private to you). Before anything else works:

1. Download `best_EfficientNetB0_FastKAN.pth` from your Google Drive
   (`MyDrive/best_EfficientNetB0_FastKAN.pth`).
2. Place it inside `artifacts/` so the path is exactly:
   `artifacts/best_EfficientNetB0_FastKAN.pth`
3. You can delete `artifacts/PUT_MODEL_HERE.txt` once the real file is there.

## 1. Fast pre-deployment check

```bash
python -m pip install -r requirements.txt
python smoke_test.py
```

Expected result: `SMOKE TEST PASSED` and a prediction summary line.

## 2. Optional local launch

```bash
streamlit run app.py
```

Open the local URL Streamlit prints, upload a test image, click **Predict**.

## 3. Deploy on Streamlit Community Cloud

1. Publish this folder to a **public** GitHub repository.
   - Because `best_EfficientNetB0_FastKAN.pth` may be larger than GitHub's
     100 MB soft limit for plain files, if it is close to or above that
     size, use **Git LFS**:
     ```bash
     git lfs install
     git lfs track "*.pth"
     git add .gitattributes
     ```
     Then add/commit/push normally. If the file is well under 100 MB, plain
     Git is fine and LFS is not required.
2. Sign in to [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. Click **Create app** → **Yup, I have an app**.
4. Select the repository, branch `main`, and entrypoint `app.py`.
5. In **Advanced settings**, select Python 3.11 or 3.12.
6. Click **Deploy** and watch the build logs.
7. Open the public URL and upload one test image from the NTHU-DDD test
   set to confirm the pipeline works end to end.

## 4. Verify the public demo

- Open the public URL in an incognito/private window.
- Upload a known-good NTHU-DDD-style image and confirm a confident,
  correct-looking prediction.
- Upload an ordinary phone selfie and confirm the app still runs cleanly
  (even if the predicted class is not the "expected" one — see limitation
  below) rather than crashing.
- Save the public URL and one screenshot of a working result.

## Important limits

This model is heavier than a typical scikit-learn tabular demo (a CNN
backbone with PyTorch). Streamlit Community Cloud's free CPU tier can run
it, but expect the first request after idling to be slower (cold start)
while the model loads into memory. If hosting ever becomes unreliable,
fall back to the tested Colab notebook and a saved screenshot.

## Known limitation — read before testing with your own photos

This model was trained on **NTHU-DDD**, a near-infrared, close-up driver
camera dataset. Ordinary daytime phone or webcam photos look visually very
different to the model (color vs. IR, lighting, framing, camera sensor),
so predictions on such photos can be unreliable even though the model
scores ~98.78% accuracy on its own held-out test set. This is a documented
dataset/domain limitation, not a bug in the code, and it is also shown
directly in the app's UI.

## Safety

Educational demo. Do not use this app for real safety-critical
(e.g. actual driving) decisions.
