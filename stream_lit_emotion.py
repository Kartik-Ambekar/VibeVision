import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Load pre-trained Haar cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load your trained emotion model
emotion_model = load_model('emotion_model_1000.keras')

# Emotion labels
emotion_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Streamlit UI
st.title("Real-Time Emotion Recognition")
st.write("This application captures live video and predicts the emotions of detected faces.")

run = st.checkbox('Start Video')
FRAME_WINDOW = st.image([])

# Start video capture
capture = cv2.VideoCapture(0)

while run:
    ret, frame = capture.read()
    
    if not ret:
        st.write("Error accessing webcam.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Extract the region of interest (ROI) from the grayscale image
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray_resized = cv2.resize(roi_gray, (48, 48))
        roi_gray_normalized = roi_gray_resized / 255.0
        roi_gray_reshaped = np.reshape(roi_gray_normalized, (1, 48, 48, 1))

        # Make the emotion prediction
        emotion_prediction = emotion_model.predict(roi_gray_reshaped)
        max_index = np.argmax(emotion_prediction)
        emotion_label = emotion_list[max_index]
        confidence = emotion_prediction[0][max_index] * 100

        # Display the emotion label and confidence percentage
        label_text = f"{emotion_label}: {confidence:.2f}%"
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame in Streamlit
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Release the video capture object
capture.release()
