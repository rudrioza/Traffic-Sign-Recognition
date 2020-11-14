# Traffic-Sign-Recognition

A Deep Neural Network to do traffic sign recognition

A lot has happened since MobilEye developed the first commercially deployed traffic sign recognition system in collaboration with Continental AG for the BMW-7 series vehicles. Quite a few vehicles have used this technology since. 
Establishing a reliable Traffic Sign Classification mechanism is a major step in our journey towards building semi-autonomous/autonomous driving systems.
This post intends to explain an approach to solve the problem of traffic sign classification and I intend to show how easy it is, to build, train and deploy a deep learning network for traffic sign classification.

#Highlights of this approach

The traffic sign dataset that we will be working on is GTSRB — German Traffic Signs. 
The approach used is deep learning.
The type of neural network used is a Convolutional Neural Network (CNN) paired with a linear classifier.
Python is the language used to program this.
The complete source code can be found here

Download and Visualize



* Use pandas and matplotlib along with the SignNames.csv to visualize data
Pre-Process Data



* Pre-processing techniques include 1. Centering around mean globally 2. Locally centering the image around the mean 3. Normalizing using Standard Deviation 4. Use Histogram Equalization
Training Data needs to be Scaled and Normalized , Extended , Also Augmented , Balanced
It is sufficient to scale and normalize the validation & test data
Scaling and Normalization

CLAHE ( Contrast Limited Adaptive Histogram Equalization ) for normalizing the image dataset as per recommendations from reviewer. It is observed that the CLAHE normalization itself scales the image so there is no need to scale the image separately
Types of augmentation techniques & rationale

We want our network to be invariant to scale , rotations, color and lighting conditions . Hence
Zoom, Rotation , Shear , Brightness Distrubance , Gaussian noise , color inversion etc.
