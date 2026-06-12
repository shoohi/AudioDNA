"""AudioDNA — Streamlit app.

Upload a game sound effect to see its spectrogram, predicted category,
and the 5 most similar sounds in the library; or browse the library.

Run with:
    streamlit run app.py
"""

import io

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import CATEGORIES, PROJECT_ROOT, SAMPLE_RATE
from src.features import TRIM_DB, features_from_audio
from src.similarity import SimilarityIndex

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

st.set_page_config(page_title="AudioDNA", page_icon="🎧", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
# cache_resource = "load once per server process, share across all sessions
# and reruns". Right for heavyweight unserializable things like the model +
# feature matrix. (cache_data, by contrast, copies its result per call —
# right for cheap transformable data, wrong for a 900-row index object.)
@st.cache_resource(show_spinner="Loading model and sound library...")
def load_index() -> SimilarityIndex:
    return SimilarityIndex.load()


def license_label(url: str) -> str:
    """Human-readable name for the common Creative Commons license URLs."""
    if not url:
        return "unknown license"
    if "publicdomain/zero" in url:
        return "CC0 (public domain)"
    for code in ("by-nc-sa", "by-nc-nd", "by-nc", "by-sa", "by-nd", "by"):
        if f"/{code}/" in url:
            return f"CC {code.upper()}"
    if "sampling" in url:
        return "CC Sampling+"
    return "see license page"


def freesound_url(freesound_id: int) -> str:
    return f"https://freesound.org/s/{freesound_id}/"


def attribution_line(row) -> str:
    """Markdown attribution: name, uploader, license, Freesound link."""
    return (
        f"**[{row['name']}]({freesound_url(row['freesound_id'])})** "
        f"by *{row['username']}* — [{license_label(row['license'])}]({row['license']})"
    )


def plot_waveform_and_melspec(y: np.ndarray, sr: int):
    """The same two views used in the EDA notebook: amplitude over time,
    and energy per mel-frequency band over time (log-scaled color)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set_title("waveform")
    axes[0].set_ylabel("amplitude")
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    img = librosa.display.specshow(
        librosa.power_to_db(S, ref=np.max), sr=sr,
        x_axis="time", y_axis="mel", ax=axes[1], cmap="magma",
    )
    axes[1].set_title("mel spectrogram")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
    fig.tight_layout()
    return fig


def render_similar_sounds(results: pd.DataFrame) -> None:
    for _, row in results.iterrows():
        left, right = st.columns([2, 3])
        with left:
            audio_path = PROJECT_ROOT / row["filepath"]
            if audio_path.exists():
                st.audio(str(audio_path))
            else:
                st.caption("audio file missing locally")
        with right:
            st.markdown(attribution_line(row))
            st.caption(
                f"category: {row['category']} · duration: {row['duration']:.1f}s "
                f"· similarity: {row['similarity']:.3f}"
            )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.title("🎧 AudioDNA")
st.caption(
    "Game sound-effect classifier and similarity search, trained on "
    "~900 Creative Commons sounds from Freesound.org."
)

try:
    index = load_index()
except FileNotFoundError:
    st.error(
        "Model bundle not found. Run the pipeline first: "
        "`python -m src.collect --all`, `python -m src.features`, "
        "`python -m src.train`."
    )
    st.stop()

bundle = index.bundle
analyze_tab, library_tab = st.tabs(["🔍 Analyze a sound", "📚 Library explorer"])

# --- Tab 1: upload & analyze ----------------------------------------------
with analyze_tab:
    uploaded = st.file_uploader(
        "Upload a sound effect (WAV or MP3, max 10 MB)", type=["wav", "mp3"]
    )

    if uploaded is None:
        st.info("Upload a short sound effect to see its AudioDNA.")
    elif uploaded.size > MAX_UPLOAD_BYTES:
        st.error("File is larger than 10 MB — please upload a shorter clip.")
    else:
        try:
            # librosa accepts a file-like object; decode exactly like the
            # training pipeline did (mono, 22050 Hz, silence-trimmed).
            y, sr = librosa.load(
                io.BytesIO(uploaded.getvalue()), sr=SAMPLE_RATE, mono=True
            )
            y, _ = librosa.effects.trim(y, top_db=TRIM_DB)
            features = features_from_audio(y, sr)  # raises if empty/too short
        except Exception as exc:
            st.error(
                f"Could not analyze this file — it may be corrupt or in an "
                f"unsupported encoding. ({exc})"
            )
        else:
            st.audio(uploaded)
            st.pyplot(plot_waveform_and_melspec(y, sr), clear_figure=True)

            # Classify: same scaler + column order as training.
            X = pd.DataFrame([features])[bundle["feature_cols"]]
            proba = bundle["model"].predict_proba(bundle["scaler"].transform(X))[0]
            probs = pd.Series(proba, index=bundle["model"].classes_)
            best = probs.idxmax()

            col_pred, col_chart = st.columns([1, 2])
            with col_pred:
                st.metric("Predicted category", best, f"{probs[best]:.0%} confidence",
                          delta_color="off")
                if probs[best] < 0.5:
                    st.caption(
                        "Low confidence — this sound sits between categories, "
                        "which is common for percussive sounds "
                        "(impact / weapon / explosion overlap acoustically)."
                    )
            with col_chart:
                st.bar_chart(probs.reindex(CATEGORIES), horizontal=True,
                             x_label="probability", y_label="")

            st.subheader("Most similar sounds in the library")
            render_similar_sounds(index.search(features, k=5))

# --- Tab 2: library explorer ------------------------------------------------
with library_tab:
    col_cat, col_page = st.columns([2, 1])
    with col_cat:
        category = st.selectbox("Category", CATEGORIES)
    subset = index.meta[index.meta["category"] == category].reset_index(drop=True)

    PAGE_SIZE = 10
    n_pages = max(1, -(-len(subset) // PAGE_SIZE))  # ceil division
    with col_page:
        page = st.number_input("Page", min_value=1, max_value=n_pages, value=1)

    st.caption(f"{len(subset)} sounds in '{category}' — page {page}/{n_pages}")
    start = (page - 1) * PAGE_SIZE
    for _, row in subset.iloc[start : start + PAGE_SIZE].iterrows():
        left, right = st.columns([2, 3])
        with left:
            audio_path = PROJECT_ROOT / row["filepath"]
            if audio_path.exists():
                st.audio(str(audio_path))
            else:
                st.caption("audio file missing locally")
        with right:
            st.markdown(attribution_line(row))
            st.caption(f"duration: {row['duration']:.1f}s")

st.divider()
st.caption(
    "All sounds are from Freesound.org under Creative Commons licenses; "
    "each entry links to its source page and license. Built as a data "
    "science portfolio project."
)
