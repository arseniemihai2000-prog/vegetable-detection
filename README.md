This project implements a Convolutional Neural Network (CNN) for vegetable classification, trained on a dataset of 21,000 images divided into 15 categories such as potato, carrot, broccoli, and others. Each class contains 1,400 images, split into 70% for training, 15% for validation, and 15% for testing.

The model architecture consists of multiple convolutional and pooling layers for feature extraction, followed by fully connected layers for classification. Training is performed using Keras/TensorFlow, with ImageDataGenerator for data augmentation to improve generalization.

The repository includes scripts for training, evaluation, and visualization of results. The trained model can be used to classify vegetable images into one of the 15 predefined classes.
