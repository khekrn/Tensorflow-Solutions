# SMS Spam-Ham classification

- Tensorflow 1.4
- Using new Dataset API for Input pipeline
- Creating Serialized model (.pbtxt) for tensorflow serving which can be deployed in cloud ml
- Local prediction using serialized file

# Instructions

- use build_vocab.py to generate vocabulary file
- train.py - Training the model
- inference.py - model inference
- embedding_model.py - Based on Representation learning for very short texts using weighted word embedding aggregation research paper
- bag_of_words.py - Simple bag of words model using tensorflow

Accuracy = 0.95
===============
![alt text](https://github.com/KishoreKarunakaran/CloudML-Serving/blob/master/text/SpamClassification/images/Accuracy.PNG)

Average Loss = 0.2246
=====================
![alt text](https://github.com/KishoreKarunakaran/CloudML-Serving/blob/master/text/SpamClassification/images/Loss.PNG)
