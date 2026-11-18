"""Coral Bleaching Detector – Streamlit demo app.

Run with:  streamlit run app.py
Needs coral_model.keras and model_info.json (from the Colab notebook) in the same folder.
"""
import io
import json
import os

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from matplotlib import colormaps

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "coral_model.keras")
INFO_PATH = os.path.join(APP_DIR, "model_info.json")
SAMPLES_DIR = os.path.join(APP_DIR, "samples")
DEFAULT_INFO = {"class_names": ["healthy", "bleached"], "threshold": 0.5, "img_size": 224, "tta": False}
UNSURE_MARGIN = 0.15   # predictions this close to the threshold are flagged as "unsure"

st.set_page_config(page_title="Coral Bleaching Detector", page_icon="🪸", layout="wide")

# ---------- Styling (works in both light and dark themes) ----------
st.markdown(
    """
    <style>
    .block-container {padding-top: 3.2rem; max-width: 1150px;}
    .hero {background: linear-gradient(120deg, #023e8a 0%, #0077b6 45%, #2a9d8f 100%);
           color: #fff; border-radius: 16px; padding: 1.4rem 1.8rem; margin-bottom: 1.2rem;}
    .hero h1 {color: #fff; margin: 0; padding: 0; font-size: 2.1rem;}
    .hero p {color: #e0f4ff; margin: .35rem 0 0 0; font-size: 1.02rem;}
    .result-card {border-radius: 14px; padding: 1.1rem 1.4rem; margin: .4rem 0 .9rem 0;
                  border-left: 8px solid; background: rgba(127,127,127,.08);}
    .result-card.healthy {border-color: #2a9d8f; background: rgba(42,157,143,.12);}
    .result-card.bleached {border-color: #e76f51; background: rgba(231,111,81,.12);}
    .result-card .verdict {font-size: 1.55rem; font-weight: 700;}
    .result-card .detail {opacity: .85; margin-top: .2rem;}
    .gauge {position: relative; height: 16px; border-radius: 8px; margin: .3rem 0 .25rem 0;
            background: linear-gradient(90deg, rgba(42,157,143,.25), rgba(231,111,81,.25));}
    .gauge .fill {position: absolute; left: 0; top: 0; bottom: 0; border-radius: 8px;}
    .gauge .marker {position: absolute; top: -5px; bottom: -5px; width: 3px; border-radius: 2px;
                    background: currentColor; opacity: .75;}
    .gauge-labels {display: flex; justify-content: space-between; font-size: .82rem; opacity: .75; margin-bottom: .7rem;}
    .tip-card {border-radius: 12px; padding: .9rem 1.1rem; background: rgba(0,119,182,.08);
               border: 1px solid rgba(0,119,182,.25); height: 100%;}
    .tip-card b {display: block; margin-bottom: .25rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Model loading and Grad-CAM ----------
@st.cache_resource(show_spinner="Loading the model…")
def load_model():
    import keras
    model = keras.models.load_model(MODEL_PATH)
    # The MobileNetV2 feature extractor is the nested model that isn't the augmentation Sequential
    base = next((layer for layer in model.layers
                 if isinstance(layer, keras.Model) and not isinstance(layer, keras.Sequential)), None)
    if base is None:
        raise RuntimeError("Could not find the MobileNetV2 feature extractor inside the saved model.")
    return model, base


def load_info():
    info = dict(DEFAULT_INFO)
    if os.path.exists(INFO_PATH):
        try:
            with open(INFO_PATH) as f:
                info.update(json.load(f))
        except (OSError, json.JSONDecodeError):
            st.sidebar.warning("`model_info.json` could not be read, so default settings are used.")
    return info


def preprocess(pil_img, size):
    """Same steps as training: RGB → shrink to 512 px → resize to size×size (pixels 0–255)."""
    import tensorflow as tf
    img = ImageOps.exif_transpose(pil_img).convert("RGB")
    img.thumbnail((512, 512))
    return tf.image.resize(np.asarray(img, dtype=np.float32), (size, size)).numpy()


def grad_cam(model, base, img_array, explain_bleached=None, threshold=0.5):
    """Returns (probability of bleached, Grad-CAM heatmap scaled 0–1) for the predicted class."""
    import tensorflow as tf
    x = model.get_layer("rescale")(tf.convert_to_tensor(img_array[None, ...], dtype=tf.float32))
    with tf.GradientTape() as tape:
        feature_maps = base(x, training=False)
        tape.watch(feature_maps)
        prob = model.get_layer("prediction")(model.get_layer("gap")(feature_maps))[:, 0]
        if explain_bleached is None:
            explain_bleached = bool(prob[0] >= threshold)
        score = prob if explain_bleached else 1 - prob
    grads = tape.gradient(score, feature_maps)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(feature_maps[0] * weights, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return float(prob[0]), cam.numpy()


def predict_prob(model, img_array, tta):
    """Probability of bleached; with TTA it averages the photo and its mirror image (as in the notebook)."""
    batch = np.stack([img_array, img_array[:, ::-1]]) if tta else img_array[None, ...]
    return float(np.mean(model(batch, training=False)))


@st.cache_data(show_spinner=False, max_entries=32)
def analyse(image_bytes, size, threshold, tta):
    """Runs the model once per photo; reruns (e.g. moving the opacity slider) reuse the result."""
    model, base = load_model()
    pil_img = Image.open(io.BytesIO(image_bytes))
    img = preprocess(pil_img, size)
    prob = predict_prob(model, img, tta) if tta else None
    raw_prob, cam = grad_cam(model, base, img,
                             explain_bleached=None if prob is None else prob >= threshold,
                             threshold=threshold)
    prob = raw_prob if prob is None else prob
    display = ImageOps.exif_transpose(pil_img).convert("RGB")
    display.thumbnail((512, 512))
    return prob, cam, np.asarray(display)


def overlay_heatmap(img_uint8, cam, alpha=0.45):
    heat = Image.fromarray(np.uint8(cam * 255)).resize(img_uint8.shape[1::-1], Image.BILINEAR)
    heat = colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    return np.uint8((1 - alpha) * img_uint8 + alpha * heat * 255)


def show_image(img):
    """Image that fills its column, on both newer (width="stretch") and older Streamlit versions."""
    try:
        st.image(img, width="stretch")
    except Exception:
        st.image(img, use_container_width=True)


def gauge_html(prob, threshold, colour):
    return (
        f'<div class="gauge"><div class="fill" style="width:{prob:.1%};background:{colour};"></div>'
        f'<div class="marker" style="left:calc({threshold:.1%} - 1px);"></div></div>'
        f'<div class="gauge-labels"><span>0% · healthy</span><span>decision line {threshold:.0%}</span>'
        f'<span>bleached · 100%</span></div>'
    )


def sample_label(filename):
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")
    icon = {"healthy": "🟢", "bleached": "⚪"}.get(parts[0], "🖼️")
    return f"{icon} {parts[0].capitalize()} – example {parts[-1]}" if len(parts) > 1 else f"{icon} {stem}"


# ---------- Sidebar: the model, then the science ----------
info = load_info()
with st.sidebar:
    st.header("🧪 Model scorecard")
    metrics = info.get("test_metrics")
    if metrics:
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
        c2.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
        c1.metric("Bleached recall", f"{metrics.get('bleached_recall', 0):.1%}")
        c2.metric("Bleached precision", f"{metrics.get('bleached_precision', 0):.1%}")
        used = info.get("images_used", {})
        split = info.get("split", {})
        st.caption(
            f"Dataset of {used.get('total', '?')} photos: {split.get('train', '?')} for training, "
            f"{split.get('validation', '?')} for validation and {split.get('test', '?')} held back for "
            "testing. These scores come from the test photos, which the model never saw during training."
        )
        baseline = info.get("baseline_colour_only")
        if baseline:
            st.caption(f"For comparison, a simple colour-only rule scores {baseline.get('accuracy', 0):.1%} accuracy.")
    else:
        st.caption("Test scores appear here once `model_info.json` from the notebook is next to `app.py`.")
    st.caption(f"Decision line: **{info['threshold']:.0%}** probability of bleaching"
               + (" · mirror-image averaging on" if info.get("tta") else ""))

    with st.expander("🪸 What is coral bleaching?"):
        st.write(
            "Corals get their colour and most of their food from microscopic algae (zooxanthellae) "
            "living in their tissue. When seawater stays unusually warm for weeks, the coral expels "
            "the algae and its white skeleton shows through. A bleached coral can recover if the water "
            "cools, but long heat stress starves it."
        )
    with st.expander("🌡️ Link to climate change"):
        st.write(
            "The ocean has absorbed more than 90% of the extra heat trapped by greenhouse gases, "
            "so marine heatwaves and mass bleaching are becoming more frequent. Reefs support "
            "fisheries, protect coastlines and drive tourism. In India they are found in the Gulf of "
            "Mannar, Gulf of Kutch, Lakshadweep and the Andaman & Nicobar Islands."
        )
    with st.expander("🧠 How this model works"):
        st.write(
            "A MobileNetV2 convolutional neural network, pre-trained on ImageNet and fine-tuned on "
            "real photos of healthy and bleached corals (transfer learning). Grad-CAM shows which "
            "parts of the photo drove the decision."
        )
    with st.expander("⚠️ Limitations"):
        st.write(
            "Classroom model trained on a small set of internet photos. It cannot tell bleached coral "
            "from dead coral or naturally pale species, and it labels any photo, even one with no coral in it."
        )


# ---------- Main page ----------
st.markdown(
    '<div class="hero"><h1>🪸 Coral Bleaching Detector</h1>'
    "<p>Upload or take a photo of coral. The model decides whether it looks <b>healthy</b> or "
    "<b>bleached</b>, and shows which parts of the photo it based its decision on.</p></div>",
    unsafe_allow_html=True,
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file `coral_model.keras` not found. Run the Colab notebook, unzip `coral_outputs.zip` "
        "and put `coral_model.keras`, `model_info.json` and the `samples` folder next to `app.py`."
    )
    st.stop()

try:
    load_model()
except Exception as e:  # usually a TensorFlow/Keras version mismatch with Colab
    tf_needed = info.get("tensorflow_version")
    st.error(
        "The model could not be loaded. "
        + (f"It was trained with TensorFlow {tf_needed}; install the same version with "
           f"`pip install tensorflow=={tf_needed}` and restart the app." if tf_needed else
           "Check that your TensorFlow version matches the one used in Colab.")
    )
    with st.expander("Error details"):
        st.code(str(e))
    st.stop()

sample_files = sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))) \
    if os.path.isdir(SAMPLES_DIR) else []
options = ["📁 Upload a photo", "📷 Take a photo"] + (["🖼️ Try a sample photo"] if sample_files else [])
source = st.radio("Choose an input", options, horizontal=True, label_visibility="collapsed")

image_bytes = None
true_label = None
if source == options[0]:
    uploaded = st.file_uploader("Coral photo (JPG, PNG or WebP)", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        image_bytes = uploaded.getvalue()
elif source == options[1]:
    shot = st.camera_input("Point the camera at a coral photo or specimen")
    if shot:
        image_bytes = shot.getvalue()
else:
    choice = st.selectbox("Sample photo (from the test set the model never trained on)", sample_files,
                          format_func=sample_label)
    with open(os.path.join(SAMPLES_DIR, choice), "rb") as f:
        image_bytes = f.read()
    true_label = choice.split("_")[0]   # sample files are named healthy_… or bleached_…

if image_bytes is None:
    st.write("")
    t1, t2, t3 = st.columns(3)
    t1.markdown('<div class="tip-card"><b>📸 Fill the frame</b>Get close so coral covers most of the photo, '
                "not water, sand or divers.</div>", unsafe_allow_html=True)
    t2.markdown('<div class="tip-card"><b>💡 Clear and sharp</b>Avoid very blurry, dark or heavily blue-tinted '
                "shots where the coral's colour is hard to see.</div>", unsafe_allow_html=True)
    t3.markdown('<div class="tip-card"><b>🔍 Check the heatmap</b>A trustworthy result focuses on the coral '
                "itself. If it looks at the background, be cautious.</div>", unsafe_allow_html=True)
    st.stop()

size = int(info.get("img_size", 224))
threshold = float(info.get("threshold", 0.5))
try:
    with st.spinner("Analysing the photo…"):
        prob_bleached, cam, display_arr = analyse(image_bytes, size, threshold, bool(info.get("tta", False)))
except (UnidentifiedImageError, OSError, ValueError):
    st.error("That file couldn't be read as an image. Please try a different JPG, PNG or WebP photo.")
    st.stop()

is_bleached = prob_bleached >= threshold
confidence = prob_bleached if is_bleached else 1 - prob_bleached
colour = "#e76f51" if is_bleached else "#2a9d8f"
verdict = "⚠️ Bleached coral detected" if is_bleached else "✅ Coral looks healthy"

st.markdown(
    f'<div class="result-card {"bleached" if is_bleached else "healthy"}">'
    f'<div class="verdict">{verdict}</div>'
    f'<div class="detail">Confidence <b>{confidence:.0%}</b> · Probability of bleaching '
    f"<b>{prob_bleached:.1%}</b></div></div>",
    unsafe_allow_html=True,
)
st.markdown(gauge_html(prob_bleached, threshold, colour), unsafe_allow_html=True)

if true_label in ("healthy", "bleached"):
    correct = (true_label == "bleached") == is_bleached
    st.caption(f"{'✅' if correct else '❌'} True label from the dataset: **{true_label}**: the model got this one "
               f"**{'right' if correct else 'wrong'}**.")
if abs(prob_bleached - threshold) < UNSURE_MARGIN:
    st.info("🤔 The model is unsure. The photo may be unclear, partly bleached, or not show coral at all.")

st.write("")
col1, col2 = st.columns(2, gap="medium")
with col1:
    st.subheader("Your photo")
    show_image(display_arr)
with col2:
    st.subheader("Where the model looked")
    alpha = st.session_state.get("heat_alpha", 0.45)
    small_cam = np.asarray(Image.fromarray(np.uint8(cam * 255)).resize(display_arr.shape[1::-1],
                                                                       Image.BILINEAR)) / 255.0
    overlay = overlay_heatmap(display_arr, small_cam, alpha)
    show_image(overlay)
    st.slider("Heatmap strength", 0.0, 0.9, 0.45, 0.05, key="heat_alpha",
              help="Slide left to see more of the photo, right to see more of the heatmap.")
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    st.download_button("⬇️ Download heatmap", buf.getvalue(), file_name="coral_gradcam.png", mime="image/png")

with st.expander("How to read the heatmap"):
    st.write(
        "Red and yellow areas influenced the decision most; blue areas barely mattered. "
        "A trustworthy prediction focuses on the coral itself, not on the water, sand or divers. "
        "If the heatmap focuses on the background, treat the result with caution."
    )
