# Image Classification Using EfficientNet

## Overview
This project focuses on image classification using **EfficientNet** as the backbone model. The goal is to classify images into different categories with high accuracy. The model was trained on a labeled dataset and optimized using transfer learning.

## Dataset
The [dataset](https://www.kaggle.com/datasets/obulisainaren/multi-cancer/data) consists of multiple image classes, each representing a specific category. Images were preprocessed, resized, and augmented to improve the model’s performance. The dataset was split into training, validation, and testing sets to ensure generalization.

## Model Choice
**EfficientNet** was chosen over other deep learning models due to its efficiency in balancing accuracy and computational cost. It scales depth, width, and resolution optimally, making it well-suited for image classification tasks.

## Outcome
The trained model achieves an accuracy of **98%**, demonstrating its effectiveness in classifying images correctly. The model is saved in `.h5` format for easy deployment and testing.

## How the Dataset Was Used
1. **Preprocessing** – Images were resized to match EfficientNet’s input size, normalized, and augmented.
2. **Training** – The model was trained using the preprocessed dataset, leveraging transfer learning and fine-tuning.
3. **Evaluation** – Performance was assessed using accuracy and loss metrics on the validation dataset.
4. **Testing** – The trained model was tested on new images to ensure reliable predictions.

## Usage
Clone this repository and use the saved model to classify new images.

