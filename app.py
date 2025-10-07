#importing the necessary libraries
import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
#load your model
model = load_model("your_model.h5")  
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
st.set_page_config(page_title="Face Mask Detection", page_icon="", layout="wide")
st.title("Face Mask Detection App")
st.write("Upload an image or use the camera to detect whether a mask is worn or not.")
st.sidebar.title("Options")
option = st.sidebar.radio("Choose input type:", ("Upload Image", "Use Webcam"))
def detect_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)

    for (x, y, w, h) in faces:
        face_img = image[y:y + h, x:x + w]
        resized = cv2.resize(face_img, (128, 128)) / 255.0
        reshaped = np.reshape(resized, (1, 128, 128, 3))
        mask_prob = model.predict(reshaped)[0][0]

        if mask_prob > 0.5:
            label = "Mask Detected"
            color = (255, 0, 0)  # Blue
        else:
            label = "No Mask Detected"
            color = (0, 0, 255)  # Red

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return image
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file).convert("RGB"))
        st.image(image, caption="Original Image", use_column_width=True)
        st.write("Processing...")

        result_img = detect_mask(image.copy())
        st.image(result_img, caption="Result", use_column_width=True)
elif option == "Use Webcam":
    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)
    while run:
        ret, frame = camera.read()
        if not ret:
            st.warning("Failed to access the camera.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_img = detect_mask(frame)
        FRAME_WINDOW.image(result_img)

    camera.release()
