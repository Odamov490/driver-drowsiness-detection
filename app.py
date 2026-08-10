from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from src.inference import load_model_bundle, predict_image

ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "artifacts" / "best_EfficientNetB0_FastKAN.pth"

# =========================================================
# Translations
# =========================================================

T = {
    "uz": {
        "lang_name": "O'zbekcha",
        "page_title": "Haydovchi uyquchanligini aniqlash",
        "subtitle": "EfficientNetB0 + FastKAN modeli asosida",
        "upload_label": "Yuz rasmini yuklang",
        "upload_help": "JPG, JPEG yoki PNG formatidagi rasm",
        "uploaded_caption": "Yuklangan rasm",
        "predict_btn": "Tahlil qilish",
        "result_title": "Natija",
        "status": "Holat",
        "confidence": "Ishonchlilik",
        "prob_title": "Klasslar bo'yicha ehtimollik",
        "drowsy": "Uyqusiragan",
        "not_drowsy": "Uyg'oq",
        "model_load_error": "Model faylini yuklab bo'lmadi.",
        "limitation_title": "Muhim eslatma",
        "limitation_body": (
            "Model NTHU-DDD dataseti (infraqizil kamera, yaqin masofadagi "
            "haydovchi yuzi) asosida o'qitilgan. Oddiy telefon yoki veb-kamera "
            "rasmlari modelga tanish emas, shuning uchun bunday rasmlarda natija "
            "noto'g'ri bo'lishi mumkin. Bu — datasetning cheklovi, dastur xatosi emas."
        ),
        "footer": "Ta'lim maqsadidagi namoyish. Haqiqiy xavfsizlik qarorlari uchun ishlatmang.",
        "classes": {
            "notdrowsy": "Uyg'oq",
            "sleepyCombination": "Uyquchan holat",
            "slowBlinkWithNodding": "Sekin pirpirash / bosh irg'ash",
            "yawning": "Esnash",
        },
    },
    "en": {
        "lang_name": "English",
        "page_title": "Driver Drowsiness Detection",
        "subtitle": "Powered by EfficientNetB0 + FastKAN",
        "upload_label": "Upload a face image",
        "upload_help": "Image in JPG, JPEG or PNG format",
        "uploaded_caption": "Uploaded image",
        "predict_btn": "Analyze",
        "result_title": "Result",
        "status": "Status",
        "confidence": "Confidence",
        "prob_title": "Probability by class",
        "drowsy": "Drowsy",
        "not_drowsy": "Alert",
        "model_load_error": "Could not load the model file.",
        "limitation_title": "Important note",
        "limitation_body": (
            "The model was trained on the NTHU-DDD dataset (near-infrared camera, "
            "close-up driver face). Ordinary phone or webcam photos are unfamiliar "
            "to the model, so results on such images may be unreliable. This is a "
            "dataset limitation, not a bug."
        ),
        "footer": "Educational demo. Do not use for real safety-critical decisions.",
        "classes": {
            "notdrowsy": "Alert",
            "sleepyCombination": "Sleepy state",
            "slowBlinkWithNodding": "Slow blink / nodding",
            "yawning": "Yawning",
        },
    },
    "ru": {
        "lang_name": "Русский",
        "page_title": "Определение сонливости водителя",
        "subtitle": "На основе модели EfficientNetB0 + FastKAN",
        "upload_label": "Загрузите изображение лица",
        "upload_help": "Изображение в формате JPG, JPEG или PNG",
        "uploaded_caption": "Загруженное изображение",
        "predict_btn": "Анализировать",
        "result_title": "Результат",
        "status": "Состояние",
        "confidence": "Уверенность",
        "prob_title": "Вероятность по классам",
        "drowsy": "Сонливость",
        "not_drowsy": "Бодрый",
        "model_load_error": "Не удалось загрузить файл модели.",
        "limitation_title": "Важное примечание",
        "limitation_body": (
            "Модель обучена на датасете NTHU-DDD (инфракрасная камера, лицо "
            "водителя крупным планом). Обычные фото с телефона или веб-камеры "
            "незнакомы модели, поэтому результаты на таких изображениях могут "
            "быть недостоверными. Это ограничение датасета, а не ошибка."
        ),
        "footer": "Учебная демонстрация. Не используйте для реальных решений безопасности.",
        "classes": {
            "notdrowsy": "Бодрый",
            "sleepyCombination": "Сонное состояние",
            "slowBlinkWithNodding": "Медленное моргание / кивание",
            "yawning": "Зевота",
        },
    },
}

# =========================================================
# Page config + styles
# =========================================================

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="centered",
)

st.markdown(
    """
    <style>
    /* Umumiy fon — Telegram Web'dagi kabi ochiq ko'k gradient */
    .stApp {
        background: linear-gradient(180deg, #E9F1F7 0%, #DCE9F2 45%, #CFE1EC 100%);
    }

    /* Markaziy "karta" — oq fon, yumaloq burchak, yengil soya */
    .block-container {
        max-width: 720px;
        padding: 2.2rem 2.4rem 2.4rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
        background: #ffffff;
        border-radius: 18px;
        box-shadow: 0 8px 30px rgba(0, 40, 80, 0.08), 0 1px 3px rgba(0, 40, 80, 0.06);
    }

    h1 {
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #17212B;
    }

    .app-subtitle {
        color: #6b7280;
        font-size: 1.02rem;
        margin-top: -0.6rem;
        margin-bottom: 1.6rem;
    }

    /* Telegram ko'k rangidagi tugma */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        height: 3rem;
        background: linear-gradient(180deg, #2AABEE 0%, #229ED9 100%);
        color: white;
        border: none;
        box-shadow: 0 2px 8px rgba(34, 158, 217, 0.35);
    }
    .stButton > button:hover {
        background: linear-gradient(180deg, #229ED9 0%, #1E8FC4 100%);
        color: white;
    }

    /* Fayl yuklash maydoni — Telegram xabar pufakchasiga o'xshash */
    div[data-testid="stFileUploaderDropzone"] {
        border-radius: 14px;
        border: 1.5px dashed #B7D4E8;
        background: #F5FAFD;
    }

    div[data-testid="stMetricValue"] { font-size: 1.7rem; }

    /* Expander (Muhim eslatma) — pufakcha ko'rinishi */
    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #E3ECF2;
        background: #F7FAFC;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    return load_model_bundle(ARTIFACT_PATH)


# =========================================================
# Language selector
# =========================================================

_, lang_col = st.columns([3, 1])
with lang_col:
    lang = st.selectbox(
        "Language",
        options=["uz", "en", "ru"],
        format_func=lambda c: T[c]["lang_name"],
        label_visibility="collapsed",
    )

t = T[lang]

# =========================================================
# Header
# =========================================================

st.title(t["page_title"])
st.markdown(f'<div class="app-subtitle">{t["subtitle"]}</div>', unsafe_allow_html=True)

try:
    model_bundle = get_model_bundle()
except FileNotFoundError:
    st.error(t["model_load_error"])
    st.stop()

# =========================================================
# Upload + predict
# =========================================================

uploaded_file = st.file_uploader(
    t["upload_label"],
    type=["jpg", "jpeg", "png"],
    help=t["upload_help"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption=t["uploaded_caption"], use_container_width=True)

    if st.button(t["predict_btn"], use_container_width=True):
        result = predict_image(image, bundle=model_bundle)

        is_drowsy = result["top_class"] != "notdrowsy"
        status_text = t["drowsy"] if is_drowsy else t["not_drowsy"]
        class_text = t["classes"].get(result["top_class"], result["top_class"])

        st.divider()
        st.subheader(t["result_title"])

        c1, c2 = st.columns(2)
        c1.metric(t["status"], status_text)
        c2.metric(t["confidence"], f"{result['confidence']:.1%}")

        summary = f"{class_text} — {result['confidence']:.1%}"
        if is_drowsy:
            st.warning(summary)
        else:
            st.success(summary)

        st.caption(t["prob_title"])
        localized_probs = {
            t["classes"].get(k, k): v for k, v in result["probabilities"].items()
        }
        st.bar_chart(localized_probs)

# =========================================================
# Limitation note + footer
# =========================================================

with st.expander(t["limitation_title"]):
    st.write(t["limitation_body"])

st.caption(t["footer"])