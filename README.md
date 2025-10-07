# Introduction
This project is used to detect whether a person is wearing a mask or not. The color used for the mask is blue. One can change the color by using the RGB values. 

# Technologies used
Python, OpenCV, Convolutional Neural Networks.

## Overview
The Face Mask Detection System is a computer vision and deep learning-based project designed to identify whether a person is wearing a face mask or not in real time using a webcam or image input.
It combines image processing, face detection, and deep learning techniques to ensure accurate recognition and classification.

## How it works
Face Detection: The system first detects human faces from the live video stream or image using OpenCV’s Haar Cascade classifier or a Deep Neural Network (DNN) model.
Mask Classification: Each detected face is passed into a Convolutional Neural Network (CNN) model. The CNN analyzes the image and predicts whether the person is wearing a mask or not wearing a mask.
Output Display: The program draws a rectangle around the face. A label (e.g., “Mask Detected” in blue or “No Mask Detected” in red) is displayed on the video feed in real time.

## Core Algorithm
Face Detection: Haar Cascade Classifier
Model Type: Convolutional Neural Network (CNN)
Training Data: Images of people with and without masks
Loss Function: Binary Cross-Entropy
Optimizer: Adam
Activation Functions: ReLU (hidden layers), Sigmoid (output layer)

## Applications
Public surveillance systems
Workplace safety monitoring
Smart entry systems (offices, malls, airports)
Healthcare monitoring
