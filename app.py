
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Paddy Crop Disease Detection",
    page_icon="🌾",
    layout="centered",
)

MODEL_PATH = Path("models/paddy_disease_model.keras")
CLASSES_PATH = Path("models/class_names.json")


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)
    return model, class_names


st.title("🌾 Paddy Crop Disease Detection")
st.write(
    "Upload a paddy leaf image to get an experimental "
    "AI-based disease classification."
)

if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.error("Model files are missing. Please train the model first.")
    st.stop()

try:
    model, class_names = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a paddy leaf image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded leaf image", width="stretch")

        resized_image = image.resize((128, 128))
        image_array = np.asarray(resized_image, dtype=np.float32)
        image_batch = np.expand_dims(image_array, axis=0)

        if st.button("Detect Disease", type="primary"):
            with st.spinner("Analyzing image..."):
                probabilities = model.predict(image_batch, verbose=0)[0]

            predicted_index = int(np.argmax(probabilities))
            predicted_class = class_names[predicted_index]
            score = float(probabilities[predicted_index])

            st.subheader("Prediction Result")
            st.success(f"Predicted class: {predicted_class}")
            st.metric("Model confidence score", f"{score * 100:.2f}%")

            st.write("Scores for all classes")
            for name, probability in sorted(
                zip(class_names, probabilities),
                key=lambda item: item[1],
                reverse=True,
            ):
                st.write(f"{name}: {probability * 100:.2f}%")

            st.caption(
                "This is an experimental model. Prediction scores are not "
                "proof of disease, and the model may make mistakes. "
                "Consult an agricultural expert before treatment decisions."
            )

    except Exception as error:
        st.error(f"Could not process the image: {error}")
