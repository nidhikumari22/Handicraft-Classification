import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import json

from tensorflow.keras.applications.efficientnet import preprocess_input

# =========================
# LOAD CLASS NAMES
# =========================
with open("class_names.json") as f:
    class_names = json.load(f)

# =========================
# STREAMLIT UI
# =========================
st.title("Handicraft Classification System")
st.write("Upload an image to classify the handicraft")

uploaded_file = st.file_uploader(
    "Upload Handicraft Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# LOAD MODEL (SAVEDMODEL)
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("efficientnet_best_handicraft_model.keras")

model = load_model()

# =========================
# PREDICTION
# =========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))

    st.image(img, caption="Uploaded Image", width=400)

    # Convert to array
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    st.success(f"✅ Predicted Class: **{class_names[class_index]}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")

    # Optional: show all probabilities
    st.subheader("Class Probabilities")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {prediction[0][i]*100:.2f}%")
