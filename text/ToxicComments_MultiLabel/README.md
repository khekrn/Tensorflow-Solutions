# Kaggle Toxic Comment Classification Challenge

More on [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

- Tensorflow 1.5
- Using new Dataset API for Input pipeline
- Creating Serialized model (.pbtxt) for tensorflow serving which can be deployed in cloud ml
- Local prediction using serialized file

# Instructions

- Download the dataset from [here](https://drive.google.com/open?id=1cPPoxId_UaIvEl19YJmNcjdErNm4Y7Ek)
- Extract the zip file to dataset folder.
- use build_vocab.py to generate vocabulary file
- train.py - Training the model
- inference.py - model inference
- custom_model.py - Based on fasttext model.

Accuracy = 0.95
===============
![alt text](https://github.com/KishoreKarunakaran/CloudML-Serving/blob/master/text/ToxicComments_MultiLabel/images/Accuracy.PNG)

Average Loss = 0.045
=====================
![alt text](https://github.com/KishoreKarunakaran/CloudML-Serving/blob/master/text/ToxicComments_MultiLabel/images/Loss.PNG)
