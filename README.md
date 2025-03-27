# face-mask-detection
import cv2
import numpy as np
import tensorflow as tf

# Load pre-trained model
mask_model = tf.keras.models.load_model("mask_detector.h5")

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
vs = cv2.VideoCapture(0)

while True:
    ret, frame = vs.read()
    frame = cv2.flip(frame, 1)  # Flip webcam for better display
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = np.expand_dims(face, axis=0) / 255.0  # Normalize

        mask_prob = mask_model.predict(face)[0][0]

        if mask_prob > 0.5:
            label = "No Mask Detected"
            color = (255, 0, 0)
        else:
            label = "Mask Detected"
            color = (0, 0, 255)

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()
