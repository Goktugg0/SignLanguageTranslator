# Hand Sign Recognition & Translator

This project uses computer vision and deep learning to recognize hand gestures/signs and translate them into text live.
It is built with Python, OpevCV, Mediapipe, TensorFlow/Keras, and scikit-learn. 

**Anyone can train their own model, get their own data and use it as their liking even though they don't know how to code!**

## Project Structure
SignLanguageTranslator/<br />
│ <br />
├── handDetector.py # Script to detect hands live and allow user to save every hand sign they want <br />
├── label_encoder.pkl # Encodes sign labels for training/inference <br />
├── model.keras # Trained Keras model for sign recognition <br />
├── scaler.pkl # Scaler for preprocessing feature data <br />
├── sign_data.csv # CSV file containing processed features for training <br />
├── train.py # Script to train the sign recognition model and saves it <br />
├── translator.py # Script to translate recognized signs into text <br />
├── requirements.txt # Project dependencies <br />
└── README.md # Project documentation <br />
For further information and detail see pydocs.

## Usage
**1. Collect hand sign data**  
   
  Run the hand detector script to record hand landmarks:
  ```
   python handDetector.py
  ```
  Collect data by pressing "s" on your keyboard to save a few signs while you do the sign in front of the camera. (20-30 saves for each sign would be sufficient)
  ![](gifs/handDetector.gif)

**2. Train the model**

  Use your collected data to train a new model: 
  ```
   python train.py
  ```
  The code will train and save the model in the most accurate way and print out the accuracy of the created model.
  ![](gifs/train.gif)

**3. Translate signs in real-time**

  Recognize and convert hand signs into text: 
  ```
   python translate.py
  ```
  This script will open a window that will translate the signs that you trained by typing them on the screen as well as with its certainity.
  ![](gifs/translator.gif)


## Dependencies
- Python 3.10-3.11 (Note this project might not work optimally in Python 3.12 and above)
- numpy
- keras-nightly
- scikit-learn
- pandas
- opencv-python
- tensorflow
- mediapipe
- protobuf <br />

For exact versions please see requirements.txt

## Licences
This project is licensed under the MIT License. See LICENSE for details.
## Acknoledgement

Mediapipe
 for hand tracking and landmark detection.

OpenCV
 for computer vision utilities.

TensorFlow/Keras for deep learning model training and inference.

Inspiration from sign language recognition gloves in MIT and FreeCodeCamp Keras with TensorFlow and Advanced Computer Vision Courses.

Special thanks to the open-source community for datasets, tools, and guides.

