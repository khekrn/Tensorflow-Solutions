# Multichannel Variable-Size Convolution for Sentence Classification
[Research Paper](https://arxiv.org/abs/1603.04513)

- Tensorflow 1.8
- Using new Dataset API for Input pipeline
- mcnn_model.py which contains implementation of DCNN research paper.
- Creating Serialized model (.pbtxt) for tensorflow serving which can be deployed in cloud ml

# Note

- Currently the code doesn't use pretrained embedding

# Instructions

- Download the dataset from [here](https://drive.google.com/open?id=1by4tC8qrAte8o5pXR2vTG6YMe-33c7eS)
- Extract the zip file to dataset folder.
- use build_vocab.py to generate vocabulary file
- train.py - Training the model
- Download the model file [here](https://drive.google.com/open?id=1pbTjOuEc6jBrJPGWrxyxJWIOZsYLcNxg)

Model
=====
![alt text](https://github.com/KishoreKarunakaran/CloudML-Serving/blob/master/research/MCNN/images/mvcnn.png)

Accuracy = 0.84
===============
![alt text](https://github.com/KishoreKarunakaran/CloudML-Serving/blob/master/research/MCNN/images/GRAPH.png)

Average Loss = 0.64
=====================
![alt text](https://github.com/KishoreKarunakaran/CloudML-Serving/blob/master/research/MCNN/images/LOSS.PNG)
