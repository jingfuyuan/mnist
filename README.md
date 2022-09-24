# Digit recognization - experiment with the MNIST dataset  

## Introduction  
This is a project of my MITx course **Machine Learning with Python - From Linear Models to Deep Learning**.  

The MNIST (Mixed National Institute of Standards and Technology) database is a large database of handwritten digits that is commonly used for training various image processing systems. The digits were collected from among Census Bureau employees and high school students. The database contains 60,000 training digits and 10,000 testing digits, all of which have been size-normalized and centered in a fixed-size image of 28 × 28 pixels. In this project, I tested many methods with this dataset for the task of classifying these images into the correct digits.

This project also includes a more challenging part, that is to train a neural network to recognize multiple digits in an image.

In part I, I tested linear regression, SVM classification, and softmax regression. In part II, applied neural networks to this task.

## What I did
- implemented linear regression using close form solution and found it's inadequate for this task
- tested scikit-learn's SVM for binary classification and multiclass classification
- implemented softmax regression using gradient descent
- implemented kernelized softmax regression. Because the kernel matrix is huge, this is computationally demanding. I found the computation can be accelerated by using the `CuPy` libary and running on Google Colab with GPU 
- experimented with different kernel functions, such as polynomial and RBF, and tested different hyperparameters
- implemented a basic neural network
- used deep learning framework `PyTorch` to perform the digit recognization task
- tested basic full-connected neural netwrok and convolutional neural network
- constructed neural network to recognize multiple digits in an image
- used GPU of Google Colab to accelerate the computation