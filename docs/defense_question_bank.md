# Defense Question Bank — Driver Drowsiness Detection

## Answer pattern
Direct answer → exact evidence → limitation or next step.
If unsure, state what must be verified instead of guessing.

| # | Likely defense question | Short answer | Exact evidence reference | Limitation / follow-up |
|---|---|---|---|---|
| 1 | What real problem does your project solve and who uses the output? | Detects whether a driver shows drowsiness signs (and which type) from a face image, to support driver-monitoring and road safety. | `README.md` "Problem statement"; notebook Cell 2. | It is a decision-support demo, not a certified safety system. |
| 2 | Why is your dataset suitable, and what is its biggest data risk? | NTHU-DDD Multi-Class is a purpose-built drowsiness dataset with 66,521 labeled frames across four states. | Notebook Cell 6 output (counts). | Biggest risk: frames come from videos, so a random split can leak near-identical frames across train/test — noted as a limitation. |
| 3 | What baseline did you use and why did you choose the final model? | Backbone EfficientNetB0 (ImageNet-pretrained) with a FastKAN head; chosen for a strong lightweight backbone plus an expressive, low-parameter classifier. | Notebook Cell 11 (model definition), Cell 13 (training). | A plain-head baseline comparison is available in the corrected notebook for a fuller ablation. |
| 4 | Where is the final unseen-data evaluation, and why is it valid? | On the held-out test split of 9,979 images, using accuracy/precision/recall/F1 + confusion matrix. | Notebook Cell 15 output. | The split is per-frame; a subject-independent split is the stronger next step. |
| 5 | Show one concrete model failure or edge case. What does it tell you? | On ordinary phone selfies (non-infrared), the model can misclassify a clearly drowsy face as notdrowsy. | Live app upload of a phone photo; in-app "Known limitation" note. | Tells us there is a domain gap; a face-crop + domain-matched input would help. |
| 6 | Show the end-to-end demo route from raw input to output. | Upload image → Predict → class + confidence + probability chart. | `https://driver-drowsiness-detection-1997.streamlit.app`; `app.py` + `src/inference.py`. | Cold start reloads the model on first request. |
| 7 | How can another person reproduce the main demo from your repository? | Clone repo, `pip install -r requirements.txt`, run `smoke_test.py`, then `streamlit run app.py`; weights ship via Git LFS. | `README.md` run steps; `requirements.txt`; `smoke_test.py`. | Requires Git LFS to pull the 120 MB model. |
| 8 | What is the most important limitation or Responsible AI risk? | The model is tuned to NTHU-DDD near-infrared imagery and must not be used for real safety-critical decisions; it can be unreliable on out-of-distribution images. | In-app "Known limitation" expander; README "Responsible AI considerations". | Mitigation: domain adaptation + subject-independent validation. |

## Self-work rehearsal question
- Question chosen: #4 — "Where is the final unseen-data evaluation, and why is it valid?"
- My ≤60 sec answer: "The final evaluation is in the notebook's test-set cell (Cell 15). After training, I run the best checkpoint on the 9,979-image test split that was held out from training and print accuracy, precision, recall, F1, and a confusion matrix. It's valid because those images were not used to update the weights; the one caveat I state openly is that the split is per-frame, so a subject-independent split would make the estimate even stronger."
- Evidence I used: notebook Cell 15 output block.
- What was weak / what I must verify: be ready to explain the per-frame split honestly if asked to push further.
