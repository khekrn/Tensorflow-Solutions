# Imdb Sentiment Analysis

- Tensorflow 1.4
- Using new Dataset API for Input pipeline
- Creating Serialized model (.pbtxt) for tensorflow serving which can be deployed in cloud ml
- Local prediction using serialized file

# Instructions

- Download dataset from [here](https://drive.google.com/open?id=1by4tC8qrAte8o5pXR2vTG6YMe-33c7eS)
- use build_vocab.py to generate vocabulary file
- train.py - Training the model
- inference.py - model inference
- cnn_model.py - Applying Convolution1D for imdb sentiment analysis

# TODO
- Improve the model

Accuracy = 0.86 and F1-Score = 0.86
===============
![alt text](https://github.com/KishoreKarunakaran/CloudML-Serving/blob/master/text/imdb_cnn/images/Accuracy.PNG)

Average Loss = 0.5681
=====================
![alt text](https://github.com/KishoreKarunakaran/CloudML-Serving/blob/master/text/imdb_cnn/images/Loss.PNG)
