# From Pixels to Letters: A High-Accuracy CPU-Real-Time American Sign Language Detection Pipeline
---
Jonas Rheiner<sup>a, 1</sup> , Daniel Kerger<sup>a, 2</sup>, Matthias Drüppel<sup>b, 3</sup>

<sup>a</sup> Center for Advanced Studies, Baden-Württemberg Cooperative State University (DHBW), Bildungscampus 13, 74076, Heilbronn, Germany

<sup>b</sup> Computer Science Department - Center for Artificial Intelligence, Baden-Württemberg Cooperative State University (DHBW), Lerchenstraße
1, 70174, Stuttgart, Germany

ORCID(s): 0009-0005-3112-8634 (J. Rheiner); 0000-0003-3064-1637 (D. Kerger); 0000-0002-6375-7743 (M. Drüppel)

Email: <sup>1</sup>jonas.rheiner@hpe.com (J. Rheiner) <sup>2</sup>daniel.kerger@hpe.com (D. Kerger) <sup>3</sup>matthias.drueppel@dhbw-stuttgart.de (M. Drüppel)

## Abstract

We introduce a CPU-real-time American Sign Language (ASL) recognition system designed to
bridge communication barriers between the deaf community and the broader public. Our multi-
step pipeline includes preprocessing, a hand detection stage, and a classification model using a
MobileNetV3 convolutional neural network backbone followed by a classification head. We train and
evaluate our model using a combined dataset of 252k labelled images from two distinct ASL datasets.
This increases generalization on unseen data and strengthens our evaluation. We employ a two-step
training: The backbone is initialized through transfer learning and frozen for the initial training of the
head. A second training phase with lower learning rate and unfrozen weights yields an exceptional test
accuracy of 99.95% and >99.93% on the two datasets - setting new benchmarks for ASL detection.
With an CPU-inference time under 500 milliseconds, it ensures real-time performance on affordable
hardware. We propose a straightforward method to determine the amount of data needed for validation
and testing and to quantify the remaining statistical error. For this we calculate accuracy as a function
of validation set size, and thus ensure sufficient data is allocated for evaluation. Model interpretability
is enhanced using Gradient-weighted Class Activation Mapping (Grad-CAM), which provides visual
explanations by highlighting key image regions influencing predictions. This transparency fosters trust
and improves user understanding of the system’s decisions. Our system sets new benchmarks in ASL
gesture recognition by closing the accuracy gap of state-of-the-art solutions, while offering broad
applicability through CPU-real-time inference and interpretability of our predictions.


## Demo

![Demo](demo.gif)

## Overview

This project implements a framework to detect American Sign Language (ASL) signs using deep learning. The framework is based on the [MobileNetV2](https://arxiv.org/abs/1801.04381) architecture.

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
