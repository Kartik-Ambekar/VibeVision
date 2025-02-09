import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained Haar cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load your trained emotion model
emotion_model = load_model('emotion_model.keras')

# Emotion labels
emotion_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start video capture
capture = cv2.VideoCapture(0)

while True:
    ret, image = capture.read()

    if not ret:
        print("Error")
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        
        # Extract the region of interest (ROI) from the grayscale image
        roi_gray = gray[y:y + h, x:x + w]
        
        # Resize the ROI to the expected input size of the emotion model
        roi_gray_resized = cv2.resize(roi_gray, (48, 48))
        
        # Normalize the pixel values to [0, 1]
        roi_gray_normalized = roi_gray_resized / 255.0
        
        # Reshape the image for the model (batch size, height, width, channels)
        roi_gray_reshaped = np.reshape(roi_gray_normalized, (1, 48, 48, 1))  # Change here: keep 1 channel

        # Make the emotion prediction
        emotion_prediction = emotion_model.predict(roi_gray_reshaped)
        
        # Get the predicted emotion label and confidence
        max_index = np.argmax(emotion_prediction)
        emotion_label = emotion_list[max_index]
        confidence = emotion_prediction[0][max_index] * 100  # Convert to percentage
        
        # Display the emotion label and confidence percentage above the face rectangle
        label_text = f"{emotion_label}: {confidence:.2f}%"
        cv2.putText(image, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('image', image)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # ESC key to exit
        break

print("Success in Facial capture.")
capture.release()
cv2.destroyAllWindows()
