# From Pixels to Letters: Building a High-Accuracy, \red{CPU-Real-Time} \red{American Sign Language} Detection Pipeline
---
Jonas Rheiner<sup>a</sup> , Daniel Kerger<sup>a</sup>, Matthias Drüppel<sup>b, 1</sup>

<sup>a</sup> Center for Advanced Studies, Baden-Württemberg Cooperative State University (DHBW), Bildungscampus 13, 74076, Heilbronn, Germany

<sup>b</sup> Computer Science Department - Center for Artificial Intelligence, Baden-Württemberg Cooperative State University (DHBW), Lerchenstraße
1, 70174, Stuttgart, Germany

ORCID(s): 0009-0005-3112-8634 (J. Rheiner); 0000-0003-3064-1637 (D. Kerger); 0000-0002-6375-7743 (M. Drüppel)

Email: <sup>1</sup>matthias.drueppel@dhbw-stuttgart.de (M. Drüppel)

## Abstract

We introduce a real-time American Sign Language (ASL) recognition system \red{bridging} communication barriers between the deaf community and the wider public. Our multi-step pipeline includes preprocessing, a hand detection stage, and a classification model. The model employs a MobileNetV3 convolutional neural network backbone with a classification head fine-tuned on the ASL Alphabet dataset. \red{We improve efficiency of ASL gestures detection by leveraging transfer learning yielding exceptional accuracy and minimal computational demands}. The pipeline achieves a test accuracy of 99.81\% - a new benchmark for ASL detection. With an inference time under 500 milliseconds without GPU acceleration, it ensures real-time performance on affordable hardware, enabling broad applicability. Our preprocessing pipeline and hand detection mechanism support model performance across diverse environments. The labelled datasets consist of static images of ASL hand signs representing each letter in the alphabet and three utility classes. To determine the appropriate amount of data for validation and testing, we propose a straightforward method to quantify the resulting statistical error. For this we calculate accuracy as a function of validation set size, and thus ensure sufficient data is allocated for reliable evaluation. Model interpretability is enhanced using Gradient-weighted Class Activation Mapping (Grad-CAM), which provides visual explanations by highlighting key image regions influencing predictions. This transparency fosters trust and improves user understanding of the system’s decisions. Our system sets a new benchmark in ASL gesture recognition by maximizing accuracy, while being efficient, and interpretable. By enabling seamless, real-time communication, it promotes greater inclusivity for the deaf and hard-of-hearing communities.

## Overview

This project implements a framework to detect American Sign Language (ASL) signs using deep learning. The framework is based on the [MobileNetV2](https://arxiv.org/abs/1801.04381) architecture and uses the [ASL Alphabet dataset](https://www.kaggle.com/grassknoted/asl-alphabet) from Kaggle. The dataset contains 87,000 images of 200x200 pixels, each representing a letter of the ASL alphabet.

The framework is implemented in Python using the [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) libraries. It consists of the following steps:

1. Camera Input: The framework captures video frames from the camera in real-time.
2. Hand Detection: The framework uses a pre-trained [Hand Detection model](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html) to detect the hand in the video frames. If a hand is detected, the image is cropped to focus on the hand region to improve the sign detection accuracy. If no hand is detected, no prediction is made.
3. Sign Detection: The cropped image is passed through the MobileNetV2 model to predict the ASL sign. The model outputs a probability distribution over the 29 classes (A-Z and space) and the sign with the highest probability is selected as the prediction.
4. Display: The predicted sign is displayed on the video frame along with the probability of the prediction.

## Requirements
- Python 3.10
- TensorFlow 2.15
- All other dependencies are listed in `requirements.txt`

## Usage
To start the sign detection framework, run the following command: `python app.py`. A window will open showing the camera input with the predicted sign and probability displayed on the screen. To close the application, close the window.
