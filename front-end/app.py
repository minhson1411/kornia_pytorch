import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Kornia Face Blur App", page_icon="🧊")

st.title("Face Blur with Kornia")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Original Image")

    if st.button("Transform"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://localhost:8000/blur-face/", files=files)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            st.image(img, caption="Blurred Faces")
        else:
            st.error("The uploaded file is not a valid image. Please another image.")
