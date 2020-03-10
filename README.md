# Segmentation of Radial Artery USG Images using Convolutional Neural Network for Artery Access Insertion
Struggling with new stuff that called Deep Learning, lots of fun will smash my brain hahaha

## The Data

Primary data was used that consist of pre-processed image on **'./data/train/image'** and labeled images for training on **'/data/train/label'**.
Almost forgot, testing images provided on **'./data/test'**, testing path will also be the directory of predicted images (result of the project).

## The Model

Inspired from [Olaf Ronneberger, U-Net Architecture](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
The flow of architecture devided by two main processes, Extraction and Expansion until finaly yield one fully connected layer.
