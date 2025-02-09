# Real-Time Emotion Recognition using Streamlit and OpenCV

## Overview
This project is a **real-time emotion recognition application** that uses a webcam feed to detect faces and classify emotions. It is built with **OpenCV**, **TensorFlow/Keras**, and **Streamlit** for the user interface.

## Features
- **Real-time face detection** using OpenCV's Haar cascades.
- **Emotion recognition** using a pre-trained deep learning model.
- **Interactive UI** powered by Streamlit.
- **Live webcam feed processing** with real-time emotion classification.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Kartik-Ambekar/emotion-recognition.git
cd emotion-recognition
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Required Files
- Place the **Haar Cascade XML file** (`haarcascade_frontalface_default.xml`) in the project directory.
- Ensure the **pre-trained emotion model** (`emotion_model_1000.keras`) is available in the project folder.

## Running the Application
Execute the following command in the project directory:
```bash
streamlit run stream_lit_emotion.py
```

This will start a **local web application**, where you can enable your webcam and see real-time emotion predictions.

## How It Works
1. **Face Detection**: OpenCV's Haar cascade classifier detects faces in the webcam feed.
2. **Preprocessing**: Detected faces are converted to grayscale, resized to 48x48 pixels, and normalized.
3. **Emotion Classification**: The pre-trained model predicts the emotion label with a confidence score.
4. **Result Display**: The detected emotion is displayed in the Streamlit UI along with the confidence percentage.

## Emotion Classes
The model classifies emotions into the following categories:
- Angry 😠
- Disgust 🤢
- Fear 😨
- Happy 😀
- Sad 😢
- Surprise 😲
- Neutral 😐

## Dependencies
The project requires the following Python libraries:
```txt
opencv-python
numpy
tensorflow
streamlit
```

## Customization
- **Model Training**: You can replace `emotion_model_1000.keras` with a custom-trained model.
- **UI Improvements**: Modify `Streamlit` components to enhance the user interface.
- **Additional Features**: Implement **multi-face detection**, **data logging**, or **graphical analysis**.

## Troubleshooting
### Webcam Not Detected
Ensure your webcam is properly connected and not in use by another application.

### Model Not Found Error
Check if `emotion_model_1000.keras` is in the project directory.

### Streamlit Not Found Error
Ensure you have installed all dependencies using:
```bash 
pip install -r requirements.txt
```

## Future Enhancements
- Add **audio-based emotion recognition**.
- Improve **accuracy** by fine-tuning the deep learning model.
- Deploy the application online using **Streamlit Cloud or AWS**.

## License
This project is **open-source** and available under the [MIT License](LICENSE).

## Author
Developed by **Kartik Ambekar**. Contributions are welcome!
