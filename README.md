# From Pixels to Letters: Building a High-Accuracy, Real-Time ASL Gesture Detection Pipeline
---
Jonas Rheiner<sup>a</sup> , Daniel Kerger<sup>a</sup>, Matthias Drüppel<sup>b, 1</sup>

<sup>a</sup> Center for Advanced Studies, Baden-Württemberg Cooperative State University (DHBW), Bildungscampus 13, 74076, Heilbronn, Germany

<sup>b</sup> Computer Science Department - Center for Artificial Intelligence, Baden-Württemberg Cooperative State University (DHBW), Lerchenstraße
1, 70174, Stuttgart, Germany

ORCID(s): 0009-0005-3112-8634 (J. Rheiner); 0000-0003-3064-1637 (D. Kerger); 0000-0002-6375-7743 (M. Drüppel)

Email: <sup>1</sup>matthias.drueppel@dhbw-stuttgart.de (M. Drüppel)

## Abstract

Sign language is essential for the deaf and hard-of-hearing communities as it allows for effective communication and inclusivity. However, outside of these communities, communication through sign language is often limited by a lack of widespread understanding and proficiency among the general population. To bridge this gap, we develop a prototype for real-time detection and understanding of American Sign Language (ASL). In this paper, we build a multi-step classification pipeline. The core of this pipeline is a deep learning image classification model trained through transfer learning. Our model achieves a 99.81% accuracy score on the test set, outperforming state-of-the-art models. To enable real time applications without GPU acceleration, we achieve an inference time of less than 500 milliseconds per input frame. The used dataset consists of static images of ASL hand signs representing each letter in the alphabet and three utility classes. Due to the transfer learning approach, we are able to train the model using little computational resources compared to training a similar classifier from scratch. Furthermore, we utilize the visualization technique Gradient-weighted Class Activation Mapping (Grad-CAM) to examine whether the model has learned meaningful visual features from the training data. Our findings indicate that the model is capable of robustly identifying and distinguishing between different hand signs, even if the input image consists of more than a single hand gesture.

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
