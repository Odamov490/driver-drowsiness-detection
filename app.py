from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from src.inference import load_model_bundle, predict_image

ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "artifacts" / "best_EfficientNetB0_FastKAN.pth"

# =========================================================
# Class metadata
# =========================================================

CLASS_ORDER = ["notdrowsy", "sleepyCombination", "slowBlinkWithNodding", "yawning"]
DROWSY_CLASSES = {"sleepyCombination", "slowBlinkWithNodding", "yawning"}

CLASS_ICONS = {
    "notdrowsy": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>',
    "sleepyCombination": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>',
    "slowBlinkWithNodding": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7"/><path d="M2 12s3 7 10 7 10-7 10-7"/></svg>',
    "yawning": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M8 14a4 4 0 0 0 8 0Z"/><path d="M9 9h.01M15 9h.01"/></svg>',
}

# =========================================================
# Translations
# =========================================================

T = {
    "uz": {
        "lang_name": "UZ",
        "brand": "Driver Drowsiness AI",
        "brand_sub": "AI asosidagi haydovchi kuzatuv tizimi",
        "hero_title": "Haydovchi uyquchanligini aniqlash",
        "hero_sub": "Haydovchi hushyorligini AI orqali tahlil qilish",
        "model_badge": "Model tayyor",
        "model_arch": "EfficientNetB0 + FastKAN",
        "upload_title": "Haydovchi rasmini yuklang",
        "upload_hint": "Bu yerga tashlang yoki fayl tanlang",
        "upload_formats": "JPG • JPEG • PNG",
        "how_title": "Qanday ishlaydi",
        "how_1": "Haydovchi rasmini yuklang",
        "how_2": "Tahlil tugmasini bosing",
        "how_3": "AI natijasini ko'ring",
        "img_info": "Rasm ma'lumoti",
        "img_name": "Fayl nomi",
        "img_format": "Format",
        "img_res": "O'lcham",
        "uploaded_caption": "Yuklangan rasm",
        "analyze_btn": "Haydovchini tahlil qilish",
        "analyzing": "Haydovchi tahlil qilinmoqda...",
        "inference": "Model ishlamoqda...",
        "result_title": "TAHLIL NATIJASI",
        "status_alert": "HUSHYOR",
        "status_drowsy": "UYQUCHAN",
        "desc_alert": "Haydovchi hushyor ko'rinadi",
        "desc_drowsy": "Uyquchanlik belgilari aniqlandi",
        "confidence": "Ishonchlilik",
        "conf_high": "Yuqori ishonch",
        "conf_med": "O'rtacha ishonch",
        "conf_low": "Past ishonch",
        "prob_title": "Klasslar bo'yicha ehtimollik",
        "limitation_title": "Muhim cheklov",
        "limitation_body": (
            "Model NTHU-DDD dataseti (infraqizil kamera, yaqin masofadagi haydovchi "
            "yuzi) asosida o'qitilgan. Oddiy telefon yoki veb-kamera rasmlarida natija "
            "noto'g'ri bo'lishi mumkin. Bu — datasetning cheklovi, dastur xatosi emas."
        ),
        "footer_edu": "Faqat ta'lim maqsadidagi namoyish",
        "model_error_title": "Model mavjud emas",
        "model_error_body": "AI modelini yuklab bo'lmadi. Model fayli mavjudligini tekshiring.",
        "img_error": "Rasmni o'qib bo'lmadi. Iltimos, JPG, JPEG yoki PNG formatidagi to'g'ri rasm yuklang.",
        "classes": {
            "notdrowsy": "Hushyor",
            "sleepyCombination": "Uyquchan holat",
            "slowBlinkWithNodding": "Sekin pirpirash / bosh irg'ash",
            "yawning": "Esnash",
        },
    },
    "en": {
        "lang_name": "EN",
        "brand": "Driver Drowsiness AI",
        "brand_sub": "AI-powered driver monitoring system",
        "hero_title": "Driver Drowsiness Detection",
        "hero_sub": "AI-powered analysis of driver alertness",
        "model_badge": "Model ready",
        "model_arch": "EfficientNetB0 + FastKAN",
        "upload_title": "Upload driver image",
        "upload_hint": "Drag & drop or browse files",
        "upload_formats": "JPG • JPEG • PNG",
        "how_title": "How it works",
        "how_1": "Upload a driver image",
        "how_2": "Click Analyze",
        "how_3": "Review AI prediction",
        "img_info": "Image information",
        "img_name": "File name",
        "img_format": "Format",
        "img_res": "Resolution",
        "uploaded_caption": "Uploaded image",
        "analyze_btn": "Analyze Driver",
        "analyzing": "Analyzing driver...",
        "inference": "Model inference...",
        "result_title": "ANALYSIS RESULT",
        "status_alert": "ALERT",
        "status_drowsy": "DROWSY",
        "desc_alert": "Driver appears alert",
        "desc_drowsy": "Signs of drowsiness detected",
        "confidence": "Confidence",
        "conf_high": "High confidence",
        "conf_med": "Medium confidence",
        "conf_low": "Low confidence",
        "prob_title": "Probability by class",
        "limitation_title": "Important limitation",
        "limitation_body": (
            "The model was trained on the NTHU-DDD dataset (near-infrared camera, "
            "close-up driver face). Results on ordinary phone or webcam photos may be "
            "unreliable. This is a dataset limitation, not a bug."
        ),
        "footer_edu": "Educational demonstration only",
        "model_error_title": "Model unavailable",
        "model_error_body": "The AI model could not be loaded. Check that the model file exists.",
        "img_error": "Could not read the image. Please upload a valid JPG, JPEG or PNG file.",
        "classes": {
            "notdrowsy": "Alert",
            "sleepyCombination": "Sleepy state",
            "slowBlinkWithNodding": "Slow blink / nodding",
            "yawning": "Yawning",
        },
    },
    "ru": {
        "lang_name": "RU",
        "brand": "Driver Drowsiness AI",
        "brand_sub": "Система мониторинга водителя на базе ИИ",
        "hero_title": "Определение сонливости водителя",
        "hero_sub": "Анализ бдительности водителя с помощью ИИ",
        "model_badge": "Модель готова",
        "model_arch": "EfficientNetB0 + FastKAN",
        "upload_title": "Загрузите изображение водителя",
        "upload_hint": "Перетащите или выберите файл",
        "upload_formats": "JPG • JPEG • PNG",
        "how_title": "Как это работает",
        "how_1": "Загрузите изображение водителя",
        "how_2": "Нажмите «Анализировать»",
        "how_3": "Посмотрите результат ИИ",
        "img_info": "Информация об изображении",
        "img_name": "Имя файла",
        "img_format": "Формат",
        "img_res": "Разрешение",
        "uploaded_caption": "Загруженное изображение",
        "analyze_btn": "Анализировать водителя",
        "analyzing": "Анализ водителя...",
        "inference": "Работа модели...",
        "result_title": "РЕЗУЛЬТАТ АНАЛИЗА",
        "status_alert": "БОДРЫЙ",
        "status_drowsy": "СОНЛИВОСТЬ",
        "desc_alert": "Водитель выглядит бодрым",
        "desc_drowsy": "Обнаружены признаки сонливости",
        "confidence": "Уверенность",
        "conf_high": "Высокая уверенность",
        "conf_med": "Средняя уверенность",
        "conf_low": "Низкая уверенность",
        "prob_title": "Вероятность по классам",
        "limitation_title": "Важное ограничение",
        "limitation_body": (
            "Модель обучена на датасете NTHU-DDD (инфракрасная камера, лицо водителя "
            "крупным планом). На обычных фото с телефона или веб-камеры результат может "
            "быть недостоверным. Это ограничение датасета, а не ошибка."
        ),
        "footer_edu": "Только учебная демонстрация",
        "model_error_title": "Модель недоступна",
        "model_error_body": "Не удалось загрузить модель ИИ. Проверьте наличие файла модели.",
        "img_error": "Не удалось прочитать изображение. Загрузите корректный файл JPG, JPEG или PNG.",
        "classes": {
            "notdrowsy": "Бодрый",
            "sleepyCombination": "Сонное состояние",
            "slowBlinkWithNodding": "Медленное моргание / кивание",
            "yawning": "Зевота",
        },
    },
}

# =========================================================
# Page config
# =========================================================

st.set_page_config(
    page_title="Driver Drowsiness AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# CSS (organized by section)
# =========================================================

def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ---------- Design tokens ---------- */
        :root{
            --bg:#F5F7FA; --card:#FFFFFF; --primary:#2563EB; --primary-dark:#1D4ED8;
            --success:#16A34A; --warning:#F59E0B; --danger:#DC2626;
            --text:#111827; --muted:#6B7280; --border:#E5E7EB;
        }

        /* ---------- Hide Streamlit chrome ---------- */
        #MainMenu{visibility:hidden;}
        footer{visibility:hidden;}
        header[data-testid="stHeader"]{background:transparent;height:0;}
        .stDeployButton{display:none;}
        [data-testid="stToolbar"]{display:none;}

        /* ---------- App background & container ---------- */
        .stApp{background:var(--bg);}
        .block-container{max-width:1080px;padding-top:1.4rem;padding-bottom:2rem;}
        html, body, [class*="css"]{color:var(--text);}

        /* ---------- Header ---------- */
        .ddx-header{
            display:flex;align-items:center;
            background:var(--card);border:1px solid var(--border);border-radius:16px;
            padding:16px 22px;box-shadow:0 1px 2px rgba(16,24,40,.04);height:100%;
        }
        .ddx-brand{display:flex;align-items:center;gap:13px;}
        .ddx-brand .ic{
            width:42px;height:42px;border-radius:11px;flex:0 0 42px;
            background:rgba(37,99,235,.1);display:flex;align-items:center;justify-content:center;
            color:var(--primary);
        }
        .ddx-brand .ic svg{width:22px;height:22px;}
        .ddx-brand h1{font-size:1.12rem;font-weight:700;margin:0;line-height:1.2;}
        .ddx-brand p{font-size:.82rem;color:var(--muted);margin:0;}

        /* ---------- Hero ---------- */
        .ddx-hero{
            background:var(--card);border:1px solid var(--border);border-radius:16px;
            padding:26px 28px;margin-bottom:18px;box-shadow:0 1px 2px rgba(16,24,40,.04);
        }
        .ddx-hero h2{font-size:1.5rem;font-weight:700;margin:0 0 4px;letter-spacing:-.02em;}
        .ddx-hero p{color:var(--muted);margin:0 0 16px;font-size:.98rem;}
        .ddx-meta{display:flex;gap:10px;flex-wrap:wrap;align-items:center;}
        .ddx-chip{
            display:inline-flex;align-items:center;gap:7px;font-size:.82rem;font-weight:600;
            padding:6px 12px;border-radius:999px;border:1px solid var(--border);color:var(--text);background:#fff;
        }
        .ddx-chip.arch{color:var(--primary);border-color:rgba(37,99,235,.25);background:rgba(37,99,235,.05);}
        .ddx-dot{width:8px;height:8px;border-radius:50%;background:var(--success);box-shadow:0 0 0 3px rgba(22,163,74,.15);}
        .ddx-chip.ready{color:var(--success);border-color:rgba(22,163,74,.25);background:rgba(22,163,74,.06);}

        /* ---------- Section titles ---------- */
        .ddx-sec{font-size:.78rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:6px 0 10px;}

        /* ---------- Upload card ---------- */
        .ddx-upload{
            background:var(--card);border:2px dashed #CBD5E1;border-radius:16px;
            padding:34px 20px;text-align:center;transition:.18s;margin-bottom:6px;
        }
        .ddx-upload:hover{border-color:var(--primary);background:#FBFDFF;}
        .ddx-upload .u-ic{
            width:56px;height:56px;border-radius:14px;margin:0 auto 12px;
            background:rgba(37,99,235,.08);color:var(--primary);display:flex;align-items:center;justify-content:center;
        }
        .ddx-upload .u-ic svg{width:28px;height:28px;}
        .ddx-upload h3{font-size:1.05rem;font-weight:600;margin:0 0 4px;}
        .ddx-upload p{color:var(--muted);font-size:.9rem;margin:0 0 8px;}
        .ddx-upload .fmt{font-size:.78rem;color:#94A3B8;font-weight:600;letter-spacing:.04em;}

        [data-testid="stFileUploaderDropzone"]{
            background:var(--card);border:1px solid var(--border);border-radius:12px;
        }

        /* ---------- How it works ---------- */
        .ddx-steps{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:4px;}
        .ddx-step{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:16px;}
        .ddx-step .num{
            width:26px;height:26px;border-radius:8px;background:rgba(37,99,235,.1);color:var(--primary);
            font-weight:700;font-size:.82rem;display:flex;align-items:center;justify-content:center;margin-bottom:9px;
        }
        .ddx-step p{margin:0;font-size:.9rem;color:var(--text);}

        /* ---------- Image info ---------- */
        .ddx-info{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px 20px;}
        .ddx-info .r{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #F1F5F9;font-size:.9rem;}
        .ddx-info .r:last-child{border-bottom:none;}
        .ddx-info .r .k{color:var(--muted);}
        .ddx-info .r .v{color:var(--text);font-weight:600;}

        /* ---------- Result ---------- */
        .ddx-result{
            background:var(--card);border:1px solid var(--border);border-radius:18px;
            padding:26px 28px;margin-top:14px;box-shadow:0 4px 18px rgba(16,24,40,.06);animation:rise .35s ease;
        }
        @keyframes rise{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:none;}}
        .ddx-result.alert{border-left:5px solid var(--success);}
        .ddx-result.drowsy{border-left:5px solid var(--danger);}
        .ddx-result .rlabel{font-size:.76rem;font-weight:700;letter-spacing:.1em;color:var(--muted);margin-bottom:14px;}
        .ddx-badge{display:inline-flex;align-items:center;gap:9px;font-weight:700;font-size:1.05rem;padding:9px 18px;border-radius:999px;border:1px solid transparent;}
        .ddx-badge .bdot{width:10px;height:10px;border-radius:50%;}
        .ddx-badge.alert{background:rgba(22,163,74,.1);color:var(--success);border-color:rgba(22,163,74,.25);}
        .ddx-badge.alert .bdot{background:var(--success);}
        .ddx-badge.drowsy{background:rgba(220,38,38,.09);color:var(--danger);border-color:rgba(220,38,38,.25);}
        .ddx-badge.drowsy .bdot{background:var(--danger);}
        .ddx-result .rdesc{margin:14px 0 20px;font-size:1.02rem;color:var(--text);}

        /* ---------- Confidence ---------- */
        .ddx-conf-top{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:6px;}
        .ddx-conf-top .lbl{font-size:.82rem;color:var(--muted);font-weight:600;}
        .ddx-conf-top .pct{font-size:1.8rem;font-weight:800;letter-spacing:-.02em;}
        .ddx-conf-note{font-size:.8rem;color:var(--muted);margin-top:6px;}
        .ddx-track{height:12px;background:#EEF2F7;border-radius:999px;overflow:hidden;}
        .ddx-fill{height:100%;border-radius:999px;transition:width .5s ease;}
        .fill-ok{background:var(--success);}
        .fill-bad{background:var(--danger);}

        /* ---------- Probability bars ---------- */
        .ddx-prob{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:22px 24px;margin-top:6px;}
        .ddx-prow{margin-bottom:14px;}
        .ddx-prow:last-child{margin-bottom:0;}
        .ddx-prow .top{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;}
        .ddx-prow .name{display:flex;align-items:center;gap:9px;font-size:.92rem;font-weight:600;color:var(--text);}
        .ddx-prow .name .g{width:18px;height:18px;color:var(--muted);display:inline-flex;}
        .ddx-prow .name .g svg{width:18px;height:18px;}
        .ddx-prow .val{font-variant-numeric:tabular-nums;font-weight:700;font-size:.92rem;color:var(--text);}
        .ddx-prow.lead .name{color:var(--primary);}
        .ddx-prow.lead .name .g{color:var(--primary);}
        .ddx-ptrack{height:9px;background:#EEF2F7;border-radius:999px;overflow:hidden;}
        .ddx-pfill{height:100%;border-radius:999px;background:#CBD5E1;transition:width .5s ease;}
        .ddx-prow.lead .ddx-pfill{background:var(--primary);}

        /* ---------- Limitation ---------- */
        .ddx-warn{
            background:#FFFBEB;border:1px solid rgba(245,158,11,.4);border-radius:14px;
            padding:16px 18px;margin-top:18px;display:flex;gap:12px;align-items:flex-start;
        }
        .ddx-warn .wic{color:var(--warning);flex:0 0 22px;margin-top:1px;display:inline-flex;}
        .ddx-warn .wic svg{width:22px;height:22px;}
        .ddx-warn h4{margin:0 0 4px;font-size:.92rem;font-weight:700;color:#92400E;}
        .ddx-warn p{margin:0;font-size:.88rem;color:#78530E;line-height:1.5;}

        /* ---------- Footer ---------- */
        .ddx-footer{text-align:center;color:var(--muted);font-size:.84rem;margin-top:26px;padding-top:18px;border-top:1px solid var(--border);}
        .ddx-footer b{color:var(--text);font-weight:600;}

        /* ---------- Error ---------- */
        .ddx-error{background:#FEF2F2;border:1px solid rgba(220,38,38,.35);border-radius:16px;padding:24px 26px;text-align:center;}
        .ddx-error h3{color:var(--danger);margin:0 0 6px;font-size:1.1rem;}
        .ddx-error p{color:#7F1D1D;margin:0;font-size:.92rem;}

        /* ---------- Analyze button ---------- */
        .stButton > button{
            background:var(--primary);color:#fff;border:none;border-radius:11px;
            font-weight:600;height:3rem;font-size:.98rem;transition:.18s;box-shadow:0 1px 2px rgba(37,99,235,.25);
        }
        .stButton > button:hover{background:var(--primary-dark);box-shadow:0 4px 12px rgba(37,99,235,.28);transform:translateY(-1px);}
        .stButton > button:focus{box-shadow:0 0 0 3px rgba(37,99,235,.3);}

        /* ---------- Segmented language radio ---------- */
        [data-testid="stRadio"] div[role="radiogroup"]{
            flex-direction:row;gap:4px;background:#EEF2F7;border-radius:10px;padding:4px;justify-content:flex-end;
        }
        [data-testid="stRadio"] div[role="radiogroup"] label{
            margin:0;padding:5px 14px;border-radius:7px;cursor:pointer;font-size:.85rem;font-weight:600;
        }

        /* ---------- Responsive ---------- */
        @media(max-width:820px){
            .ddx-steps{grid-template-columns:1fr;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Cached model loader (inference contract unchanged)
# =========================================================

@st.cache_resource(show_spinner=False)
def get_model_bundle():
    return load_model_bundle(ARTIFACT_PATH)


def confidence_note(t: dict, conf: float) -> str:
    if conf >= 0.85:
        return t["conf_high"]
    if conf >= 0.6:
        return t["conf_med"]
    return t["conf_low"]


# =========================================================
# App
# =========================================================

inject_css()

# ---- Header row: brand + language selector ----
head_left, head_right = st.columns([3, 1.1])
with head_right:
    lang = st.radio(
        "lang",
        options=["uz", "en", "ru"],
        format_func=lambda c: T[c]["lang_name"],
        horizontal=True,
        label_visibility="collapsed",
        key="lang",
    )
t = T[lang]

with head_left:
    st.markdown(
        f"""
        <div class="ddx-header">
          <div class="ddx-brand">
            <span class="ic">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                   stroke-linecap="round" stroke-linejoin="round">
                <path d="M5 17h14M6 17l-1.5-5.5A3 3 0 0 1 7.4 7h9.2a3 3 0 0 1 2.9 4.5L18 17"/>
                <circle cx="7.5" cy="17.5" r="1.5"/><circle cx="16.5" cy="17.5" r="1.5"/>
              </svg>
            </span>
            <div>
              <h1>{t['brand']}</h1>
              <p>{t['brand_sub']}</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ---- Load model (professional error card) ----
try:
    model_bundle = get_model_bundle()
except Exception:
    st.markdown(
        f'<div class="ddx-error"><h3>{t["model_error_title"]}</h3><p>{t["model_error_body"]}</p></div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ---- Hero ----
st.markdown(
    f"""
    <div class="ddx-hero">
      <h2>{t['hero_title']}</h2>
      <p>{t['hero_sub']}</p>
      <div class="ddx-meta">
        <span class="ddx-chip ready"><span class="ddx-dot"></span>{t['model_badge']}</span>
        <span class="ddx-chip arch">{t['model_arch']}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Upload visual + native uploader ----
st.markdown(
    f"""
    <div class="ddx-upload">
      <div class="u-ic">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <path d="M17 8l-5-5-5 5"/><path d="M12 3v12"/>
        </svg>
      </div>
      <h3>{t['upload_title']}</h3>
      <p>{t['upload_hint']}</p>
      <div class="fmt">{t['upload_formats']}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    t["upload_title"],
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

# ---- Empty state ----
if uploaded_file is None:
    st.markdown(f'<div class="ddx-sec">{t["how_title"]}</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="ddx-steps">
          <div class="ddx-step"><div class="num">1</div><p>{t['how_1']}</p></div>
          <div class="ddx-step"><div class="num">2</div><p>{t['how_2']}</p></div>
          <div class="ddx-step"><div class="num">3</div><p>{t['how_3']}</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    try:
        image = Image.open(uploaded_file)
        image.load()
    except Exception:
        st.markdown(
            f'<div class="ddx-error"><h3>{t["model_error_title"]}</h3><p>{t["img_error"]}</p></div>',
            unsafe_allow_html=True,
        )
        st.stop()

    prev_col, info_col = st.columns([1.3, 1])
    with prev_col:
        st.image(image, caption=t["uploaded_caption"], use_container_width=True)
    with info_col:
        fmt = (image.format or uploaded_file.name.split(".")[-1].upper())
        st.markdown(f'<div class="ddx-sec">{t["img_info"]}</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="ddx-info">
              <div class="r"><span class="k">{t['img_name']}</span><span class="v">{uploaded_file.name[:26]}</span></div>
              <div class="r"><span class="k">{t['img_format']}</span><span class="v">{fmt}</span></div>
              <div class="r"><span class="k">{t['img_res']}</span><span class="v">{image.size[0]} × {image.size[1]}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        analyze = st.button(t["analyze_btn"], use_container_width=True)

    if analyze:
        with st.spinner(f"{t['analyzing']}  {t['inference']}"):
            result = predict_image(image, bundle=model_bundle)

        top_class = result["top_class"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]

        is_drowsy = top_class in DROWSY_CLASSES
        status_word = t["status_drowsy"] if is_drowsy else t["status_alert"]
        status_desc = t["desc_drowsy"] if is_drowsy else t["desc_alert"]
        state_cls = "drowsy" if is_drowsy else "alert"
        fill_cls = "fill-bad" if is_drowsy else "fill-ok"
        conf_pct = confidence * 100

        st.markdown(
            f"""
            <div class="ddx-result {state_cls}">
              <div class="rlabel">{t['result_title']}</div>
              <span class="ddx-badge {state_cls}"><span class="bdot"></span>{status_word}</span>
              <div class="rdesc">{status_desc}</div>
              <div class="ddx-conf-top">
                <span class="lbl">{t['confidence']}</span>
                <span class="pct">{conf_pct:.1f}%</span>
              </div>
              <div class="ddx-track"><div class="ddx-fill {fill_cls}" style="width:{conf_pct:.1f}%;"></div></div>
              <div class="ddx-conf-note">{confidence_note(t, confidence)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        rows_html = ""
        ordered = [c for c in CLASS_ORDER if c in probabilities]
        for cls in ordered:
            p = probabilities[cls] * 100
            name = t["classes"].get(cls, cls)
            icon = CLASS_ICONS.get(cls, "")
            lead = " lead" if cls == top_class else ""
            rows_html += (
                f'<div class="ddx-prow{lead}">'
                f'<div class="top"><span class="name"><span class="g">{icon}</span>{name}</span>'
                f'<span class="val">{p:.1f}%</span></div>'
                f'<div class="ddx-ptrack"><div class="ddx-pfill" style="width:{p:.1f}%;"></div></div>'
                f'</div>'
            )
        st.markdown(
            f'<div class="ddx-sec" style="margin-top:18px">{t["prob_title"]}</div>'
            f'<div class="ddx-prob">{rows_html}</div>',
            unsafe_allow_html=True,
        )

# ---- Limitation (always visible) ----
st.markdown(
    f"""
    <div class="ddx-warn">
      <span class="wic">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
          <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z"/>
          <path d="M12 9v4"/><path d="M12 17h.01"/>
        </svg>
      </span>
      <div>
        <h4>{t['limitation_title']}</h4>
        <p>{t['limitation_body']}</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Footer ----
st.markdown(
    f"""
    <div class="ddx-footer">
      <b>Driver Drowsiness AI</b> · EfficientNetB0 + FastKAN<br>
      {t['footer_edu']} · © 2026
    </div>
    """,
    unsafe_allow_html=True,
)