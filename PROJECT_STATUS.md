# Project Status — Driver Drowsiness Detection Deployment

## Primary route
Streamlit Community Cloud deployed from a public GitHub repository.

## Fallback route
Run the same `app.py` locally, or show a saved screenshot of the working
public app (see Colab notebook results for reference test-set metrics).

## Demo contract
- Input: one uploaded face image (jpg/jpeg/png).
- Output: predicted class (`notdrowsy`, `sleepyCombination`,
  `slowBlinkWithNodding`, `yawning`), confidence, and a probability bar chart.
- Validation: missing/invalid images are rejected with a clear message.
- Entry point: `app.py`.

## Gate status
Green after all items below are confirmed:
- [ ] `artifacts/best_EfficientNetB0_FastKAN.pth` is uploaded and loads.
- [ ] `python smoke_test.py` passes.
- [ ] GitHub repository is public and contains all required files
      (including the model weights, or Git LFS is configured for them).
- [ ] Streamlit deployment completes.
- [ ] The public URL opens in an incognito/private window.
- [ ] One known-good input produces a visible result.

## Known-good input
Use a sample image from the notebook's `test_samples` (drawn from the
NTHU-DDD dataset itself) — these reliably match the model's training
distribution and produce a confident, correct prediction.

## Known limitation
The model is trained on **NTHU-DDD**, a near-infrared, close-up driver
camera dataset. Ordinary RGB phone/webcam photos are visually out of
distribution for the model (different lighting, color, framing, sensor),
so predictions on such photos may be unreliable even though test-set
accuracy is ~98.78%. This is documented in the app UI itself
("Known limitation" expander).

## Main risk
Streamlit Community Cloud has finite CPU and memory, and PyTorch + a CNN
backbone is heavier than a typical tabular scikit-learn demo. This project
is still in the "suitable but noticeably heavier" range — CPU inference on
a single image is fast (~1-2s), but repository size (model weights) and
first-load time are the main risks. If hosting becomes unreliable, fall
back to the tested Colab notebook.
