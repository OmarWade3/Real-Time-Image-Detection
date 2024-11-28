
# Real-Time Image Detection with Pre-trained CNN

This project is a part of the **"CNN Architectures in Industrial Applications" workshop** held from **November 24 to 28, 2024**, at **Universiti Teknologi Malaysia (UTM)**. It demonstrates the application of Convolutional Neural Networks (CNNs) in real-world datasets through a pre-trained ResNet18 model for multi-class image detection.

---

## Project Overview

This repository contains:
1. **Model Training Code** (`cnn_pretrained.py`): Fine-tunes a ResNet18 model to classify images into 8 classes and saves the trained model as `pre_trained_model.pt`.
2. **Real-Time Detection Code** (`real_time.py`): Uses the fine-tuned model to perform real-time image detection through a webcam.
3. **Pre-trained Model** (`pre_trained_model.pt`): The fine-tuned model ready for inference.

---

## Features
- Classifies images into 8 categories:
  - `airplane`, `car`, `cat`, `dog`, `flower`, `fruit`, `motorbike`, `person`
- Real-time image detection with confidence thresholding.
- Visualizes model performance with loss and accuracy curves.


**Note**: Press **`q`** to exit the webcam feed.

## Model Download
You can download the pre-trained model from the following link:
- [Download pre_trained_model.pt](https://drive.google.com/file/d/1924f5kDbKNe_A9gDnHimRVhwQM_jU0cD/view?usp=sharing)




