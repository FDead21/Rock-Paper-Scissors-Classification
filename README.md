# Rock Paper Scissors Image Classification

This project implements a convolutional neural network (CNN) (VGG16 architecture to be specific) to classify images of hand gestures representing rock, paper, and scissors.

## Project Overview

The model uses transfer learning with a pre-trained VGG16 network to classify images into three categories: rock, paper, and scissors. It includes data augmentation techniques to improve model performance and generalization other than that it also uses learning rate scheduler to find the optimal learning rate for the training.

## Features

- Image classification using VGG16 transfer learning
- Data augmentation for improved model performance
- Learning rate finder implementation
- Model evaluation and performance visualization
- Confusion matrix generation

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pandas

## Setup

**Clone the repository:**

git clone https://github.com/FDead21/rock-paper-scissors-classification.git

cd rock-paper-scissors-classification


**The script will:**

- Prepare and augment the data
- Create and train the model
- Evaluate the model on the validation set
- Generate visualizations of the results


**Model Architecture**

The model uses a pre-trained VGG16 network as a base, with the following modifications:

- Removal of the top (fully connected) layers
- Addition of a Global Average Pooling layer
- ADdition of a Dense layer with 256 units and ReLU activation
- Output layer with 3 units (for the 3 classes) and softmax activation


**Results**

The model achieves high accuracy on the validation set. Detailed metrics, including a confusion matrix and F1-scores for each class, are generated and saved in the results directory.


**Future Improvements**

- Experiment with different pre-trained models (e.g., ResNet, MobileNet, Inception, etc.)
- Implement cross-validation
- Try more data augmentation techniques
- Fine-tune hyperparameters
