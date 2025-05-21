# app.py

import sys
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from infer import generate_caption
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("caption-app")
logger.setLevel(logging.INFO)

#logger.info("Application started")

import streamlit as st
st.title("Image Caption Generator")
st.write("Upload an image and get an AI-generated caption.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        img_array = img_to_array(image.resize((224, 224)))
        logger.info("Calling infer.py generate_caption")
        caption = generate_caption(img_array)
        st.success("**Generated Caption:**")
        st.markdown(f"*{caption}*")
        logger.info("Caption successfully generated")

