# ImageNet Classification with Deep Convolutional Neural Networks

## Abstract 
- $6*10^7$ parameters
- 1000 classes (classification)
- 5 conv layers(some max-pooled)
- softmax used in the end
- new regularization method: dropout : to reduce overfitting

## Introduction
- classification with small datasets pretty easy now
- using a very large dataset comparitively
- compared to standard feed-forward networks, CNNs have some parameter reuse which results in a smaller footprint and lower computational costs when backpropogating only at a very minimal drop in accuracy: CNNs are easier to train

- final network: 5 convolutional layers and 3 fully connected layers

## Dataset
- 15 million high res images : 22,000 categories(for imagenet dataset)
- for the competition: fined down to 1000 categories and 1.2 million training images, 50000 validation images and 150,000 test images
- two kinds of error rates: top-1 and top-5
	- top-5: fraction of test images for which correct label is not among the top 5 labels predicted by the model(17% in this case)
	- top-1: conventional error-rate(37.5% in this case)
- downsampled original images to 256*256
- subtracted the mean activity over the training set from each pixel
	- hence trained the model on the (centered) raw RGB values of the pixels
	- note on why to scale images first found here:
		- https://stats.stackexchange.com/questions/211436/why-normalize-images-by-subtracting-datasets-image-mean-instead-of-the-current
	
## Architecture
- contains 8 learned layers ( 5 conv and 3 fc) 

### ReLU Nonlinearity:
- using the ReLU activation function instead of the sigmoid or the hyperbolic tangent:
	- ReLU : $f(x) = max(0,x)$
	- sigmoid : $f(x) = \frac{1}{1+e^{-x}}$
	- hyperbolic tan: $f(x) = tanh(x)$
- training with ReLU (non-saturating non-linear function) is much faster than training with the other two(saturating non-linear ones)

### Local Response Normalization
- read this article before proceeding:
	- https://towardsdatascience.com/difference-between-local-response-normalization-and-batch-normalization-272308c034ac
		- why normalize? : to address the exploding gradient problem
		- batch norm : https://www.youtube.com/watch?v=tNIpEZLv_eg
	- using inter-channel local response normalization here
	- read about intra-channel LRN and batch normalization in the article linked above.

### Overlapping pooling
- observations noted: difficult to overfit with overlapping pooling

## Reducing Overfitting
- two strategies:
	1. Data Augmentation: transforming images while preserving labels
	2. Dropout: during training (i.e. a forward pass and backpropogating), set the weight of a neuron to 0 with probability 0.5 but use all the weights in preditiction of test sets with the weights multiplied with 0.5: avoids overfitting and forces a neuron to learn with random combinations of other neurons each time.

 	
