Training for this project was done in the Colab notebook, not here.

This app is inference-only: it loads the already-trained weights from
artifacts/best_EfficientNetB0_FastKAN.pth and does not retrain anything.

If you need to retrain or fine-tune the model, use your Colab notebook
(Driver_Drowsiness_Detection_last.ipynb) as the source of truth, then
re-export best_EfficientNetB0_FastKAN.pth and replace the file in
artifacts/.
