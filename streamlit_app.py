import sys

# =====================================================
# KERAS / TENSORFLOW FIX
# =====================================================

sys.path.insert(0, r"C:\tf_pkg")

# =====================================================
# IMPORTS
# =====================================================
import numpy as np
import os
import json

from datetime import datetime
import streamlit as st
import h5py
import numpy as np
import plotly.graph_objects as go

from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

from keras.models import load_model

from huggingface_hub import InferenceClient

# =====================================================
# HUGGING FACE CONFIG
# =====================================================

HF_TOKEN = ""

HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"

hf_client = InferenceClient(
    model=HF_MODEL,
    token=HF_TOKEN
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Gravitational Wave Detector",
    page_icon="🌌",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

body {
    background-color: #050816;
    color: white;
}

.main {
    background-color: #050816;
}

h1, h2, h3 {
    color: #00d9ff;
}

.stButton > button {
    background-color: #00d9ff;
    color: black;
    border-radius: 12px;
    font-weight: bold;
}

.stFileUploader {
    background-color: #111827;
    padding: 20px;
    border-radius: 15px;
}

[data-testid="stMetric"] {
    background-color: #111827;
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SETTINGS
# =====================================================

WINDOW_SIZE = 1024

STEP_SIZE = 256

THRESHOLD = 0.8

MAX_SIGNAL_LENGTH = 200000

MAX_POINTS = 5000

# =====================================================
# HISTORY STORAGE
# =====================================================

HISTORY_DIR = "history"

os.makedirs(
    HISTORY_DIR,
    exist_ok=True
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_cnn_model():

    model = load_model(
        "models/cnn_model.keras"
    )

    return model


try:

    model = load_cnn_model()

except Exception as e:

    st.error(
        f"Failed to load CNN model: {e}"
    )

    st.stop()

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def normalize_signal(signal):

    signal = np.asarray(
        signal,
        dtype=np.float64
    ).ravel()

    max_abs = np.max(
        np.abs(signal)
    )

    if max_abs == 0:
        return signal

    return signal / max_abs


def reduce_noise(
    signal,
    cutoff=0.05
):

    signal = np.asarray(
        signal,
        dtype=np.float64
    ).ravel()

    if signal.size == 0:

        raise ValueError(
            "Signal is empty."
        )

    b, a = butter(
        N=4,
        Wn=cutoff,
        btype="low"
    )

    padlen = 3 * max(
        len(a),
        len(b)
    )

    if signal.size <= padlen:

        return signal

    filtered = filtfilt(
        b,
        a,
        signal
    )

    return filtered


def read_ligo_hdf5(uploaded_file):

    with h5py.File(
        uploaded_file,
        "r"
    ) as f:

        if "strain" not in f:

            raise ValueError(
                "Missing strain group."
            )

        if "Strain" not in f["strain"]:

            raise ValueError(
                "Missing Strain dataset."
            )

        strain_ds = f["strain"]["Strain"]

        strain = np.asarray(
            strain_ds[:],
            dtype=np.float64
        ).ravel()

        if strain.size == 0:

            raise ValueError(
                "Empty signal."
            )

        xspacing = float(
            strain_ds.attrs.get(
                "Xspacing",
                1.0 / 4096
            )
        )

        meta = {}

        if "meta" in f:

            for key in f["meta"].keys():

                value = f["meta"][key][()]

                if isinstance(
                    value,
                    bytes
                ):

                    value = value.decode(
                        "utf-8",
                        errors="ignore"
                    )

                meta[key] = value

    return strain, xspacing, meta


def split_windows(signal):

    windows = []

    positions = []

    for start in range(
        0,
        len(signal) - WINDOW_SIZE,
        STEP_SIZE
    ):

        end = start + WINDOW_SIZE

        chunk = signal[start:end]

        if len(chunk) != WINDOW_SIZE:
            continue

        windows.append(chunk)

        positions.append(start)

    return np.array(
        windows
    ), positions


def predict_signal(signal):

    windows, positions = split_windows(
        signal
    )

    if len(windows) == 0:

        raise ValueError(
            "No windows generated."
        )

    windows = windows.reshape(
        windows.shape[0],
        windows.shape[1],
        1
    )

    probabilities = model.predict(
        windows,
        verbose=0
    ).flatten()

    wave_windows = (
        probabilities > THRESHOLD
    )

    wave_detected = np.any(
        wave_windows
    )

    max_probability = float(
        np.max(probabilities)
    )

    return (
        wave_detected,
        max_probability,
        probabilities,
        positions
    )

# =====================================================
# AI EXPLANATION FUNCTION
# =====================================================

def generate_ai_explanation(
    meta,
    confidence,
    detected,
    fft_scale,
    max_probability
):

    detector = meta.get(
        "Detector",
        "Unknown"
    )

    utc_start = meta.get(
        "UTCstart",
        "Unknown"
    )

    duration = meta.get(
        "Duration",
        "Unknown"
    )

    detection_str = (
        "GRAVITATIONAL WAVE DETECTED"
        if detected else
        "NO GRAVITATIONAL WAVE DETECTED"
    )

    dominant_frequency = float(
    xf[np.argmax(yf)]
    )
    
    average_probability = float(
    np.mean(probabilities)
    )

    probability_std = float(
    np.std(probabilities)
    )

    peak_position = int(
    positions[np.argmax(probabilities)]
    )

    signal_energy = float(
    np.sum(normalized_clean ** 2)
    )

    prompt = f"""
Analyze this gravitational-wave detection result.

Detector: {detector}
UTC Time: {utc_start}
Duration: {duration}s

Detection Result: {detection_str}
Confidence: {confidence:.2f}%
FFT Scale: {fft_scale}
Max Probability: {max_probability:.4f}
Dominant Frequency: {dominant_frequency:.2f} Hz
Average Probability: {average_probability:.4f}
Probability Std: {probability_std:.4f}
Peak Signal Position: {peak_position}
Signal Energy: {signal_energy:.4f}

Focus MOSTLY on interpreting the uploaded signal and model output.

Explain:
- what the confidence suggests
- what the probability behavior suggests
- whether the signal appears localized or noisy
- what the FFT pattern may indicate
- whether the signal resembles chirp-like behavior
- whether noise reduction appears effective

Avoid long textbook theory.
Focus on interpreting THIS specific analysis result.
Use short markdown sections. Do not treat CNN confidence as absolute scientific confirmation.
Mention uncertainty realistically.
Avoid saying “no doubt” or “certain detection”.
"""
    response = hf_client.chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=8192,
        temperature=0.5,
    )

    return response.choices[0].message.content


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title(
    "🌌 Navigation"
)

page = st.sidebar.radio(
    "Go To",
    [
        "Home",
        "Upload & Detect",
        "Analysis History",
        "Documentation",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader(
    "⚙ Visualization Settings"
)

fft_scale = st.sidebar.selectbox(
    "FFT Scale",
    [
        "Linear",
        "Log"
    ]
)

max_frequency = st.sidebar.slider(
    "Max FFT Frequency",
    50,
    2048,
    500
)

noise_cutoff = st.sidebar.slider(
    "Noise Filter Strength",
    0.01,
    0.50,
    0.05
)

show_original = st.sidebar.checkbox(
    "Show Original Signal",
    value=True
)

show_cleaned = st.sidebar.checkbox(
    "Show Noise Reduced Signal",
    value=True
)

# =====================================================
# HOME PAGE
# =====================================================

if page == "Home":

    st.title(
        "🌌 CNN Gravitational Wave Detector"
    )

    st.markdown("""
    ### Deep Learning Based Gravitational Wave Detection

    This system:
    - Reads real LIGO HDF5 files
    - Applies noise reduction
    - Splits waveform into windows
    - Uses CNN deep learning
    - Detects gravitational-wave patterns
    """)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Model",
        "1D CNN"
    )

    c2.metric(
        "Input",
        "Waveform"
    )

    c3.metric(
        "Threshold",
        f"{THRESHOLD}"
    )

# =====================================================
# UPLOAD PAGE
# =====================================================

elif page == "Upload & Detect":

    st.title(
        "🛰 Upload Gravitational Wave File"
    )

    uploaded_file = st.file_uploader(
        "Upload .hdf5 File",
        type=["hdf5", "h5"]
    )

    if uploaded_file is not None:

        try:

            # =========================================
            # LOAD SIGNAL
            # =========================================

            strain, dt, meta = read_ligo_hdf5(
                uploaded_file
            )

            if len(strain) > MAX_SIGNAL_LENGTH:

                strain = strain[
                    :MAX_SIGNAL_LENGTH
                ]

            # =========================================
            # NOISE REDUCTION
            # =========================================

            clean_signal = reduce_noise(
                strain,
                cutoff=noise_cutoff
            )

            normalized_original = normalize_signal(
                strain
            )

            normalized_clean = normalize_signal(
                clean_signal
            )

            # =========================================
            # DISPLAY SIGNALS
            # =========================================

            display_original = normalized_original[
                :MAX_POINTS
            ]

            display_clean = normalized_clean[
                :MAX_POINTS
            ]

            st.success(
                "✅ File Uploaded Successfully"
            )

            # =========================================
            # METADATA
            # =========================================

            st.subheader(
                "🧾 File Metadata"
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Detector",
                str(
                    meta.get(
                        "Detector",
                        "Unknown"
                    )
                )
            )

            c2.metric(
                "Duration",
                str(
                    meta.get(
                        "Duration",
                        "Unknown"
                    )
                )
            )

            c3.metric(
                "UTC Start",
                str(
                    meta.get(
                        "UTCstart",
                        "Unknown"
                    )
                )
            )

            # =========================================
            # ORIGINAL SIGNAL
            # =========================================

            if show_original:

                st.subheader(
                    "📡 Original Noisy Signal"
                )

                original_fig = go.Figure()

                original_fig.add_trace(
                    go.Scatter(
                        y=display_original,
                        mode="lines",
                        name="Original Signal",
                        line=dict(
                            color="#ff4b4b"
                        )
                    )
                )

                original_fig.update_layout(
                    template="plotly_dark",
                    height=400,
                    xaxis_title="Sample Index",
                    yaxis_title="Amplitude"
                )

                st.plotly_chart(
                    original_fig,
                    use_container_width=True
                )

            # =========================================
            # CLEAN SIGNAL
            # =========================================

            if show_cleaned:

                st.subheader(
                    "🧹 Noise Reduced Signal"
                )

                clean_fig = go.Figure()

                clean_fig.add_trace(
                    go.Scatter(
                        y=display_clean,
                        mode="lines",
                        name="Clean Signal",
                        line=dict(
                            color="#00d9ff"
                        )
                    )
                )

                clean_fig.update_layout(
                    template="plotly_dark",
                    height=400,
                    xaxis_title="Sample Index",
                    yaxis_title="Amplitude"
                )

                st.plotly_chart(
                    clean_fig,
                    use_container_width=True
                )

            # =========================================
            # OVERLAY GRAPH
            # =========================================

            st.subheader(
                "⚖ Original vs Noise Reduced"
            )

            overlay_fig = go.Figure()

            overlay_fig.add_trace(
                go.Scatter(
                    y=display_original,
                    mode="lines",
                    name="Original",
                    line=dict(
                        color="#ff4b4b"
                    ),
                    opacity=0.5
                )
            )

            overlay_fig.add_trace(
                go.Scatter(
                    y=display_clean,
                    mode="lines",
                    name="Noise Reduced",
                    line=dict(
                        color="#00d9ff"
                    )
                )
            )

            overlay_fig.update_layout(
                template="plotly_dark",
                height=500,
                xaxis_title="Sample Index",
                yaxis_title="Amplitude"
            )

            st.plotly_chart(
                overlay_fig,
                use_container_width=True
            )

            # =========================================
            # FFT ANALYSIS
            # =========================================

            st.subheader(
                "🌌 Frequency Spectrum Analysis"
            )

            yf = np.abs(
                fft(display_clean)
            )

            xf = fftfreq(
                len(display_clean),
                d=dt
            )

            mask = xf >= 0

            xf = xf[mask]

            yf = yf[mask]

            freq_mask = (
                xf <= max_frequency
            )

            xf = xf[freq_mask]

            yf = yf[freq_mask]

            fft_fig = go.Figure()

            fft_fig.add_trace(
                go.Scatter(
                    x=xf,
                    y=yf,
                    mode="lines",
                    name="FFT Spectrum",
                    line=dict(
                        color="#00ff88"
                    )
                )
            )

            if fft_scale == "Log":

                fft_fig.update_layout(
                    yaxis_type="log"
                )

            fft_fig.update_layout(
                template="plotly_dark",
                height=500,
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude"
            )

            st.plotly_chart(
                fft_fig,
                use_container_width=True
            )

            # =========================================
            # CNN PREDICTION
            # =========================================

            st.subheader(
                "🚀 Detection Result"
            )

            (
                detected,
                confidence,
                probabilities,
                positions
            ) = predict_signal(
                normalized_clean
            )

            if confidence < 0.3:

                interpretation = (
                    "Likely Noise"
                )

            elif confidence < 0.6:

                interpretation = (
                    "Uncertain Signal"
                )

            elif confidence < 0.8:

                interpretation = (
                    "Possible Wave"
                )

            else:

                interpretation = (
                    "Strong Detection"
                )

            if detected:

                st.success(
                    f"""
                    ✅ GRAVITATIONAL WAVE DETECTED

                    Confidence: {confidence * 100:.2f}%

                    Interpretation: {interpretation}
                    """
                )

            else:

                st.error(
                    f"""
                    ❌ NO GRAVITATIONAL WAVE DETECTED

                    Confidence: {confidence * 100:.2f}%

                    Interpretation: {interpretation}
                    """
                )

            # =========================================
            # PROBABILITY GRAPH
            # =========================================

            st.subheader(
                "📊 Window Prediction Probabilities"
            )

            prob_fig = go.Figure()

            prob_fig.add_trace(
                go.Scatter(
                    x=positions,
                    y=probabilities,
                    mode="lines",
                    name="Wave Probability",
                    line=dict(
                        color="#7c5cff"
                    )
                )
            )

            prob_fig.add_hline(
                y=THRESHOLD,
                line_dash="dash",
                line_color="red"
            )

            prob_fig.update_layout(
                template="plotly_dark",
                height=450,
                xaxis_title="Signal Position",
                yaxis_title="Probability"
            )

            st.plotly_chart(
                prob_fig,
                use_container_width=True
            )

            # =========================================
            # AI EXPLANATION SECTION
            # =========================================

            st.markdown("---")

            st.subheader(
                "🧠 AI Scientific Interpretation"
            )

            st.markdown(
                "*Powered by Hugging Face · Qwen2.5-72B — Explaining your results in plain English*"
            )

            if HF_TOKEN == "YOUR_HF_TOKEN_HERE":

                st.warning(
                    "⚠️ Please set your Hugging Face token in the code to enable AI explanations. "
                    "Get a free token at: https://huggingface.co/settings/tokens"
                )

            else:

                with st.spinner(
                    "🤗 Generating AI explanation via Hugging Face..."
                ):

                    try:

                        ai_explanation = generate_ai_explanation(
                            meta=meta,
                            confidence=confidence * 100,
                            detected=detected,
                            fft_scale=fft_scale,
                            max_probability=np.max(probabilities)
                        )

                        st.markdown(
                            ai_explanation
                        )

                        # =========================================
                        # SAVE ANALYSIS HISTORY
                        # =========================================

                        analysis_data = {

                            "timestamp": str(datetime.now()),

                            "detector": str(
                                meta.get(
                                    "Detector",
                                    "Unknown"
                                )
                            ),

                            "utc_start": str(
                                meta.get(
                                    "UTCstart",
                                    "Unknown"
                                )
                            ),

                            "duration": str(
                                meta.get(
                                    "Duration",
                                    "Unknown"
                                )
                            ),

                            "confidence": float(confidence),
                            
                            "max_probability": float(
                                np.max(probabilities)
                            ),

                            "interpretation": interpretation,

                            "detected": bool(detected),

                            "dominant_frequency": float(
                                xf[np.argmax(yf)]
                            ),

                            "ai_explanation": ai_explanation
                        }

                        history_filename = (
                            f"{datetime.now().timestamp()}.json"
                        )

                        history_path = os.path.join(
                            HISTORY_DIR,
                            history_filename
                        )

                        with open(
                            history_path,
                            "w",
                            encoding="utf-8"
                        ) as f:

                            json.dump(
                                analysis_data,
                                f,
                                indent=4
                            )

                    except Exception as ai_err:

                        err_str = str(ai_err)

                        if "429" in err_str or "rate" in err_str.lower() or "too many" in err_str.lower():

                            st.warning(
                                "⏳ **Hugging Face Rate Limit Hit**\n\n"
                                "Free tier has been throttled. "
                                "Wait a moment and click Retry below.\n\n"
                                "Tip: Upgrade to HF Pro for higher limits → https://huggingface.co/pricing"
                            )

                        elif "401" in err_str or "unauthorized" in err_str.lower():

                            st.error(
                                "❌ **Invalid HF Token** — "
                                "Check your token at https://huggingface.co/settings/tokens "
                                "and make sure it has **Read** permission."
                            )

                        elif "503" in err_str or "loading" in err_str.lower():

                            st.info(
                                "⏳ **Model is Loading** — "
                                "Hugging Face is warming up the model (first call takes ~30s). "
                                "Click Retry in a moment."
                            )

                        else:

                            st.error(
                                f"❌ **AI Error:** {ai_err}"
                            )

                        if st.button(
                            "🔄 Retry AI Explanation",
                            key="retry_ai"
                        ):

                            st.rerun()

        except Exception as e:

            st.error(
                f"Error: {e}"
            )

# =====================================================
# ANALYSIS HISTORY PAGE
# =====================================================

elif page == "Analysis History":

    st.title(
        "🕘 Analysis History"
    )

    history_files = sorted(
        os.listdir(HISTORY_DIR),
        reverse=True
    )

    if len(history_files) == 0:

        st.info(
            "No saved analyses found."
        )

    else:

        for file_name in history_files:

            file_path = os.path.join(
                HISTORY_DIR,
                file_name
            )

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            with st.expander(
                f"{data['timestamp']} | "
                f"{data['detector']} | "
                f"{data['interpretation']}"
            ):

                st.write(
                    f"### Detector: {data['detector']}"
                )

                st.write(
                    f"UTC Start: {data['utc_start']}"
                )

                st.write(
                    f"Confidence: "
                    f"{data['confidence'] * 100:.2f}%"
                )

                st.write(
                    f"Dominant Frequency: "
                    f"{data['dominant_frequency']:.2f} Hz"
                )

                st.write(
                    f"Interpretation: "
                    f"{data['interpretation']}"
                )

                st.markdown(
                    data["ai_explanation"]
                )

                if st.button(
                    f"🗑 Delete {file_name}",
                    key=file_name
                ):

                    os.remove(file_path)

                    st.rerun()

# =====================================================
# DOCUMENTATION PAGE
# =====================================================

elif page == "Documentation":

    st.title(
        "📖 Documentation"
    )

    st.markdown("""
# 📂 Supported File Format

The system accepts:

- `.hdf5`
- `.h5`

LIGO strain waveform files from:
- GWOSC
- LIGO Open Science Center

---

# 📡 Signal Pipeline

## Raw Detector Signal
The uploaded waveform contains detector strain measurements over time.

## Noise Reduction
Low-pass filtering reduces unwanted detector noise.

## Window Analysis
The waveform is divided into overlapping windows:

- Window Size: 1024
- Step Size: 256

This allows localized waveform detection.

---

# 🧠 CNN Detection System

The CNN model analyzes waveform patterns and predicts:

- gravitational-wave probability
- chirp-like structures
- transient signal behavior

---

# 🌌 FFT Analysis

FFT converts the waveform into the frequency domain.

This helps analyze:
- dominant frequencies
- spectral peaks
- detector noise behavior

---

# 📊 Probability Graph

The probability graph shows:

- CNN confidence across signal regions
- localized waveform activity
- possible chirp positions

Higher peaks indicate regions more similar to learned gravitational-wave patterns.

---

# 🧹 Noise Reduction Interpretation

The noise-reduced waveform helps reveal:
- transient structures
- chirp-like features
- hidden waveform behavior

while suppressing random fluctuations.

---

# 🤗 AI Scientific Interpretation

The AI explanation system interprets:

- confidence behavior
- FFT characteristics
- signal localization
- chirp likelihood
- noise characteristics

using contextual signal metrics.

---

# ⚠ Limitations

This system is NOT an official LIGO detection pipeline.

Outputs should be interpreted as:
- educational
- experimental
- research-oriented

rather than confirmed astrophysical discoveries.

---

# 🔬 Recommended Datasets

Example gravitational-wave events:

- GW150914
- GW170817
- GW190521

Available from:
https://www.gw-openscience.org
""")

# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "About Project":

    st.title(
        "📚 About Project"
    )

    st.markdown("""
# 🌌 Explainable AI Gravitational-Wave Analysis System

This project is an interactive scientific AI platform designed to analyze and interpret gravitational-wave detector signals using deep learning, signal processing, and explainable AI techniques.

Unlike traditional classification-only systems, this platform combines waveform detection with contextual interpretation and scientific visualization.

---

## 🚀 Core Objectives

The system was developed to:

- Analyze real LIGO detector strain data
- Detect gravitational-wave-like waveform patterns
- Visualize noisy and filtered detector signals
- Interpret CNN predictions in human-readable form
- Help beginners understand gravitational-wave analysis

---

# 🧠 Technologies Used

## Deep Learning
- 1D Convolutional Neural Networks (CNNs)
- Sliding-window waveform inference
- Probability localization

## Signal Processing
- FFT (Fast Fourier Transform)
- Noise reduction filtering
- Signal normalization
- Frequency-domain analysis

## Scientific Data
- Real LIGO / GWOSC HDF5 datasets
- Strain waveform analysis
- Detector metadata extraction

## Explainable AI
- AI-generated contextual interpretation
- Confidence analysis
- Signal behavior explanation
- Beginner-friendly scientific summaries

---

# 🌌 What Is a Gravitational Wave?

Gravitational waves are ripples in spacetime caused by extremely energetic cosmic events such as:

- Black hole mergers
- Neutron star collisions
- Massive accelerating objects

These waves slightly stretch and compress spacetime as they travel across the universe.

LIGO detectors measure these tiny distortions using laser interferometry.

---

# 🔬 Project Workflow

## 1️⃣ Upload HDF5 Signal
The system reads real detector strain data from LIGO HDF5 files.

## 2️⃣ Noise Reduction
Signal filtering suppresses unwanted detector noise while preserving waveform structure.

## 3️⃣ Window Segmentation
The waveform is split into smaller overlapping windows for localized CNN analysis.

## 4️⃣ CNN Inference
The deep learning model predicts gravitational-wave probability for each signal region.

## 5️⃣ Frequency Analysis
FFT analysis visualizes dominant frequencies and spectral behavior.

## 6️⃣ Explainable Interpretation
AI-generated explanations analyze:
- confidence behavior
- probability localization
- chirp-like structures
- noise characteristics
- signal quality

---

# 📊 Visualization Features

- 📡 Original detector waveform
- 🧹 Noise-reduced waveform
- ⚖ Overlay comparison
- 🌌 FFT spectrum analysis
- 📈 Probability localization graphs
- 🧠 AI scientific interpretation

---

# 🎯 Research Direction

This project explores:

- Explainable AI for astrophysics
- Gravitational-wave signal interpretation
- Deep-learning robustness under noise
- Interactive scientific visualization
- Educational AI systems for astronomy

---

# ⚠ Important Scientific Note

This system is a research and educational prototype.

CNN confidence scores do NOT represent absolute astrophysical confirmation.

Scientific gravitational-wave discovery requires:
- multi-detector verification
- statistical validation
- peer review
- astrophysical parameter estimation

---

# 👨‍💻 Built With

- Python
- Streamlit
- TensorFlow / Keras
- Plotly
- SciPy
- Hugging Face Inference API
- LIGO Open Data
""")