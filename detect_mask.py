import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pygame

# Load trained model
model = load_model("mask_detector_model.h5")

# Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (128, 128))
        face_resized = face_resized / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)

        preds = model.predict(face_resized)
        prob = preds[0][0]   # Single probability output

        if prob > 0.5:
            label = f"No Mask ({prob:.2f})"
            color = (0, 0, 255)  # Green
        else:
            label = f"Mask ({1-prob:.2f})"
            color = (0, 255, 0)  # Red
            pygame.mixer.init()
            pygame.mixer.music.load("alert.mp3")
            pygame.mixer.music.play()


        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Face Mask Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()