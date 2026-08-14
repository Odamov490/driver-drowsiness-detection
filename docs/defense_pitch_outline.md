# Defense Pitch Outline — Driver Drowsiness Detection

## Five-minute route

| Time | Block | My exact line / evidence route |
|---|---|---|
| 0:00–0:30 | Opening | "My name is *[full name]*. My project is a Driver Drowsiness Detection system that classifies a driver's face image into four states to support road-safety monitoring." |
| 0:30–1:15 | User and ML task | "The user is a driver-monitoring context. Input: one face image. ML task: image classification into notdrowsy, sleepyCombination, slowBlinkWithNodding, yawning. Output: predicted class + confidence." |
| 1:15–2:10 | Data and approach | "Dataset: public NTHU-DDD Multi-Class, 66,521 images (notebook Cell 6). Preprocessing: resize to 224×224 + ImageNet normalization. Approach: EfficientNetB0 backbone with a FastKAN classification head, trained 20 epochs with class-weighted loss to handle imbalance." |
| 2:10–3:10 | Results and weakness | "On the held-out test split (9,979 images) the model reaches 98.78% accuracy with balanced per-class F1 (notebook Cell 15). Weakest class is yawning (precision 0.969). Honest limitation: the split is per-frame, and the model is tuned to NTHU-DDD's near-infrared imagery, so ordinary phone photos are harder." |
| 3:10–4:20 | Showcase and live demo | "Live app: driver-drowsiness-detection-1997.streamlit.app. I upload one image, press Predict, and it returns the class and a per-class probability chart — running the trained weights from the repo's artifacts/ folder." |
| 4:20–5:00 | Close | "Next improvement: a subject-independent (driver-based) evaluation and a face-crop front-end so everyday camera images match the training distribution better." |

## Demo route
- Showcase entry: `https://driver-drowsiness-detection-1997.streamlit.app`
- Live demo entry: same URL → "Upload a face image" → select image → "Predict"
- One real input → output example: upload an NTHU-DDD-style test frame → app shows predicted class + confidence + probability bar chart.
- Backup screenshot/link (fallback only): notebook Cell 15 output (metrics) if the live app cold-starts slowly.

## Self-work rehearsal log
- Actual duration: *[fill after rehearsal]*
- One thing to cut / clarify: *[fill after rehearsal]*
- Random defense question chosen: "What baseline did you use and why did you choose the final model?"
- My ≤60 sec answer: "EfficientNetB0 is a strong, lightweight ImageNet-pretrained backbone; I added a FastKAN head as the classifier because KAN-style layers can model non-linear feature interactions with few parameters. I trained it with class-weighted loss and selected the checkpoint with the best validation accuracy."
- Exact evidence used: notebook Cell 11 (model) + Cell 13 (training log / best-checkpoint saving).
- What I still need to verify: rehearse timing so the demo fits inside 5 minutes.
- One pitch revision before defense: lead with the live demo if the app is warm, to save time.
