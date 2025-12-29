import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

import json

with open("class_names.json") as f:
    class_names = json.load(f)

model_names = ["custom_cnn", "mobilenet", "efficientnet"]
st.title("Handicraft Classification System")


uploaded_file = st.file_uploader("Upload Handicraft Image", type=["jpg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((224,224))
    img_array = np.array(img)/255.0
    img_array = img_array.reshape(1,224,224,3)

    
    st.image(img, caption="Uploaded Image")
    
    for name in model_names:
        try:
            model_path = f"{name}_handicraft_model.h5"
            model = tf.keras.models.load_model(model_path)

            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]

            st.success(f"Prediction from model {name}: {predicted_class}")

        except Exception as e:
            st.error(f"Failed to load or predict with model {name}: {e}")