# Kaggle Consumer Finance Complaints Classification

More on [here](https://www.kaggle.com/cfpb/us-consumer-finance-complaints)

- Tensorflow 1.4
- Using new Dataset API for Input pipeline
- Creating Serialized model (.pbtxt) for tensorflow serving which can be deployed in cloud ml
- Local prediction using serialized file

# Instructions

- Download the dataset from [here](https://drive.google.com/open?id=1j9d1zyEaxVRwTm2zjOiQdJ5SxcusvZ_N)
- Extract the zip file to dataset folder.
- use build_vocab.py to generate vocabulary file
- train.py - Training the model
- inference.py - model inference
- cnn_model.py - Based on Convolutional Neural Networks for Sentence Classification - Yoon Kim Research paper.

Accuracy = 0.92
===============
![alt text](https://github.com/KishoreKarunakaran/CloudML-Serving/blob/master/text/finance-complaints/images/Accuracy.PNG)

Average Loss = 0.5
=====================
![alt text](https://github.com/KishoreKarunakaran/CloudML-Serving/blob/master/text/finance-complaints/images/Loss.PNG)
