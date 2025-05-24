# asl_hand_sign_recognition
A real-time ASL hand sign detection system using OpenCV and a CNN model. It detects hands, preprocesses the image, predicts the alphabet (A–Z), and displays the result live from webcam input. Built using TensorFlow/Keras and cvzone.

✋ Hand Sign Detection Using OpenCV, cvzone & Keras
📌 Overview
This project uses a webcam to detect hand gestures in real time and classify them into corresponding American Sign Language (ASL) alphabet letters using a pre-trained deep learning model.

⚙️ Technologies & Libraries Used
OpenCV – for image capturing and processing.

cvzone – for easy hand detection and bounding box extraction.

TensorFlow/Keras – for loading and running the trained classification model.

NumPy – for handling image arrays and transformations.

🧠 How It Works
Capture Frame
The webcam captures real-time video using cv2.VideoCapture().

Detect Hands
Using cvzone.HandTrackingModule.HandDetector, the code detects one or more hands and retrieves the bounding box around the hand.

Crop and Resize the Image
The region of interest (ROI) around the hand is cropped from the frame. This cropped image is resized to match the input size of the model (224x244 pixels in this case).

Preprocess Input

The cropped image is normalized (divided by 255) to scale pixel values between 0 and 1.

It is reshaped to the expected input shape for the model: (1, 224, 244, 3).

Prediction
The preprocessed image is passed to the Keras model using model.predict(), which returns a probability array for all possible ASL characters.

Display Output
The predicted class (highest probability index) is mapped to a label (e.g., 'A', 'B', ..., 'Z') and displayed on the frame with a bounding box around the hand.

🏷️ Model Details
Model file: Model/model2.h5

Labels file: Model/label.txt

Input size: (224, 244, 3)

Output: Predicted ASL letter (A–Z)

🖥️ User Interaction
The program continuously runs until the user presses the "q" key to quit.

During execution, the cropped image and final output frame are shown in two OpenCV windows:

ImageCrop: Cropped hand image

Image: Final output with prediction overlay

✅ Applications
Assistive technology for the hearing and speech impaired

Educational tools for learning sign language

Real-time ASL translators
