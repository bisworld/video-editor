"""
Underwater AI Video Cutter & Enhancer
Локальное Streamlit-приложение (OpenCV + MoviePy), без внешних API.

Запуск на Ubuntu:
  1. sudo apt update && sudo apt install python3-venv imagemagick
  2. python3 -m venv venv
  3. source venv/bin/activate
  4. pip install --upgrade pip && pip install -r requirements.txt
  5. streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from video_processor import process_video

st.set_page_config(
    page_title="Underwater AI Video Cutter & Enhancer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

AQUATIC_CSS = """
<style>
    html, body, [data-testid="stAppViewContainer"] {
        font-size: 18px !important;
    }
    .stApp {
        background: linear-gradient(165deg, #031a24 0%, #0a3d4a 35%, #0d7377 70%, #14919b 100%);
        color: #e8f7f8;
        font-family: 'Segoe UI', 'Ubuntu', 'Cantarell', sans-serif;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #022833 0%, #0a4a55 100%);
        border-right: 1px solid rgba(100, 220, 230, 0.25);
        min-width: 380px !important;
        width: 380px !important;
    }
    [data-testid="stSidebar"] * {
        color: #d7f3f5 !important;
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 1.55rem !important;
        margin-bottom: 0.4rem !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        font-size: 1.05rem !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] input {
        font-size: 1.05rem !important;
        min-height: 2.6rem !important;
    }
    [data-testid="stSidebar"] .stCheckbox label p {
        font-size: 1.1rem !important;
    }
    h1 {
        color: #a8f0f5 !important;
        text-shadow: 0 2px 18px rgba(0, 200, 220, 0.35);
        letter-spacing: -0.02em;
        font-size: 2.6rem !important;
        line-height: 1.2 !important;
        margin-bottom: 0.6rem !important;
    }
    h2, h3 {
        font-size: 1.6rem !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #0ea5a8, #14b8a6) !important;
        color: #022833 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 1rem 2rem !important;
        font-size: 1.25rem !important;
        min-height: 3.4rem !important;
        box-shadow: 0 6px 20px rgba(20, 184, 166, 0.35);
    }
    .stButton > button:hover {
        filter: brightness(1.08);
    }
    .stDownloadButton > button {
        background: linear-gradient(90deg, #06b6d4, #22d3ee) !important;
        color: #022833 !important;
        font-weight: 700 !important;
        width: 100%;
        border-radius: 16px !important;
        padding: 1.15rem 1.8rem !important;
        font-size: 1.35rem !important;
        min-height: 3.6rem !important;
    }
    [data-testid="stFileUploader"] {
        background: rgba(8, 55, 65, 0.55);
        border: 2px dashed rgba(94, 234, 212, 0.45);
        border-radius: 18px;
        padding: 1.5rem;
    }
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span {
        font-size: 1.15rem !important;
    }
    [data-testid="stFileUploader"] button {
        font-size: 1.1rem !important;
        padding: 0.7rem 1.2rem !important;
    }
    div[data-testid="stProgressBar"] > div {
        height: 14px !important;
        border-radius: 8px !important;
    }
    .status-card {
        background: rgba(4, 40, 50, 0.65);
        border: 1px solid rgba(94, 234, 212, 0.3);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0 1.5rem;
        font-size: 1.2rem;
    }
    .hint {
        color: #9fd8de;
        font-size: 1.25rem;
        line-height: 1.5;
        margin-bottom: 1.75rem;
    }
    .stAlert, [data-testid="stAlert"] {
        font-size: 1.15rem !important;
    }
    video {
        border-radius: 16px;
        width: 100% !important;
    }
</style>
"""
st.markdown(AQUATIC_CSS, unsafe_allow_html=True)


def _save_upload(uploaded) -> str:
    suffix = Path(uploaded.name).suffix or ".mp4"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="underwater_in_")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


with st.sidebar:
    st.markdown("## ⚙️ Настройки")
    st.caption("Все вычисления выполняются локально")

    target_duration = st.slider(
        "Финальная длина видео",
        min_value=10,
        max_value=120,
        value=30,
        step=5,
        help="Суммарная длительность итогового ролика (сек)",
    )

    resolution = st.selectbox(
        "Разрешение видео",
        options=[
            "Оригинальное",
            "1080p (1920x1080) — Горизонтальное",
            "720p (1280x720) — Горизонтальное",
            "1080x1920 (Shorts/Reels) — Вертикальное",
        ],
        index=0,
    )

    priority = st.radio(
        "Приоритет сюжета",
        options=["Баланс", "Больше рыб", "Больше пейзажей"],
        index=0,
    )

    fish_sensitivity = st.slider(
        "Чувствительность к рыбам",
        min_value=1,
        max_value=10,
        value=5,
        help="Порог детекции мелких контрастных контуров",
    )

    st.markdown("---")
    st.markdown("## ✨ Качество и фильтры")

    exclude_people = st.checkbox(
        "Исключить людей из кадра",
        value=True,
        help="YOLO person/backpack + тёмный гидрокостюм (MOG2); блок ±2 с",
    )
    enhance_color = st.checkbox(
        "Улучшить контраст и цвет под водой",
        value=True,
        help="CLAHE по каналу яркости (LAB)",
    )
    blur_filter = st.slider(
        "Фильтр мутных кадров",
        min_value=0,
        max_value=100,
        value=30,
        help="Порог Laplacian variance: ниже — кадр слишком размыт",
    )


st.title("Underwater AI Video Cutter & Enhancer 🌊")
st.markdown(
    '<p class="hint">Загрузите подводное видео — приложение отберёт лучшие моменты, '
    "отфильтрует муть и людей, улучшит цвет и соберёт короткий ролик.</p>",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Загрузить видео",
    type=["mp4", "mov", "avi"],
    help="Поддерживаются MP4, MOV, AVI",
)

if "result_path" not in st.session_state:
    st.session_state.result_path = None
if "result_bytes" not in st.session_state:
    st.session_state.result_bytes = None

run = st.button(
    "Запустить обработку",
    type="primary",
    disabled=uploaded is None,
    use_container_width=True,
)

progress_bar = st.progress(0)
status_box = st.empty()

if run and uploaded is not None:
    st.session_state.result_path = None
    st.session_state.result_bytes = None
    input_path = None
    try:
        status_box.markdown(
            '<div class="status-card">Сохранение загруженного файла…</div>',
            unsafe_allow_html=True,
        )
        input_path = _save_upload(uploaded)

        def on_progress(value: float, text: str) -> None:
            progress_bar.progress(value)
            status_box.markdown(
                f'<div class="status-card">{text}</div>',
                unsafe_allow_html=True,
            )

        out_fd, out_path = tempfile.mkstemp(suffix=".mp4", prefix="underwater_out_")
        os.close(out_fd)

        result = process_video(
            video_path=input_path,
            target_duration=target_duration,
            resolution=resolution,
            priority=priority,
            fish_sensitivity=fish_sensitivity,
            exclude_people=exclude_people,
            enhance_color=enhance_color,
            blur_filter=blur_filter,
            output_path=out_path,
            progress=on_progress,
        )

        with open(result, "rb") as f:
            st.session_state.result_bytes = f.read()
        st.session_state.result_path = result
        progress_bar.progress(1.0)
        status_box.markdown(
            '<div class="status-card">Готово!</div>',
            unsafe_allow_html=True,
        )
        st.success("Монтаж завершён — смотрите превью ниже.")
    except Exception as exc:
        progress_bar.progress(0)
        status_box.markdown(
            f'<div class="status-card">Ошибка: {exc}</div>',
            unsafe_allow_html=True,
        )
        st.error(str(exc))
    finally:
        if input_path and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except OSError:
                pass

if st.session_state.result_bytes:
    st.subheader("Результат")
    st.video(st.session_state.result_bytes)
    st.download_button(
        label="Скачать готовое видео",
        data=st.session_state.result_bytes,
        file_name="underwater_highlight.mp4",
        mime="video/mp4",
        use_container_width=True,
    )
