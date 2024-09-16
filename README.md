# Realtime Gesture Detection & Understanding of American Sign Language with Transfer Learning
---
Jonas Rheiner∗, Daniel Kerger†, Matthias Drüppel‡

Computer Science Department of the Baden-Wuerttemberg Cooperative State University (DHBW)

Heilbronn and Stuttgart, Germany

Email: ∗cas367123@cas.dhbw.de, †cas366673@cas.dhbw.de, ‡matthias.drueppel@dhbw-stuttgart.de

**Abstract:** Sign language is essential for the deaf and hardof-hearing communities as it allows for effective communication and inclusivity. However, outside of these communities, communication through sign language is often limited by a lack of widespread understanding and proficiency among the general population. To bridge this gap, we develop a prototype for realtime detection and understanding of American Sign Language (ASL). In this paper, we build a deep learning image classification model leveraging transfer learning. Our model achieves a 99.72% accuracy score, outperforming state-of-the-art models. To enable real time applications without GPU acceleration, we achieve an inference time of less than 500 milliseconds per input frame on the validation data. The underlying validation data consists of static images of ASL hand signs representing each letter in the alphabet and three utility classes. Due to the transfer learning approach, we are able to train the model using little computational resources compared to training a similar classifier from scratch. Furthermore, we utilize the visualization technique Gradientweighted Class Activation Mapping (Grad-CAM) to examine whether the model has learned meaningful visual features from the training data. Our findings indicate that the model is capable of successfully identifying and distinguishing between different hand signs, even when presented as part of a single input image. Index Terms—American Sign Language, Deep Learning, Convolutional Neural Networks, Transfer Learning
