from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from src.inference import load_model_bundle, predict_image

ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "artifacts" / "best_EfficientNetB0_FastKAN.pth"

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    """Load the saved model once for the app process."""
    return load_model_bundle(ARTIFACT_PATH)


st.title("Driver Drowsiness Detection")
st.caption("EfficientNetB0 + FastKAN — saved model -> inference function -> Streamlit wrapper")

st.info(
    "This app does not train a model. It loads a saved PyTorch model "
    "(best_EfficientNetB0_FastKAN.pth) and runs inference on one uploaded image."
)

try:
    model_bundle = get_model_bundle()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a face image (jpg, jpeg, png)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict", use_container_width=True):
        try:
            result = predict_image(image, bundle=model_bundle)
        except ValueError as exc:
            st.error(f"Input rejected: {exc}")
        else:
            st.divider()
            st.subheader("Prediction result")

            label_col, conf_col = st.columns(2)
            label_col.metric("Status", result["label"])
            conf_col.metric("Confidence", f"{result['confidence']:.1%}")

            if result["label"] == "Drowsy":
                st.warning(result["summary"])
            else:
                st.success(result["summary"])

            st.caption("Probability by class")
            st.bar_chart(result["probabilities"])

            st.caption(f"Saved artifact version: {result['model_version']}")

with st.expander("What this deployment demonstrates"):
    st.markdown(
        """
        - The model weights are committed in `artifacts/`.
        - `src/inference.py` owns model definition, validation, and prediction.
        - `app.py` is only the user-interface wrapper.
        - Streamlit Community Cloud runs the app from the GitHub repository.
        """
    )

with st.expander("Known limitation — read before testing with your own photos"):
    st.markdown(
        """
        This model was trained on the **NTHU-DDD** dataset, which consists of
        near-infrared, close-up driver camera footage. Ordinary daytime phone
        or webcam photos look visually very different to the model (color,
        lighting, framing, camera sensor), so predictions on such photos can
        be unreliable even when the model scores ~98% accuracy on its own
        test set. This is a known dataset/domain limitation, not a bug.
        """
    )

st.caption("Educational demo. Do not use this app for real safety-critical decisions.")
