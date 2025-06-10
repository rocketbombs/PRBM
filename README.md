MNIST Classification using a Restricted Boltzmann Machine (RBM) and a Gray Code Classifier

This project demonstrates a two-stage approach to image classification on the MNIST dataset. First, an unsupervised Restricted Boltzmann Machine (RBM) is trained to learn hierarchical features from the raw pixel data. These learned features are then fed into a simple, lightweight supervised classifier that uses a Gray code representation for the final digit classification.
Description

The core of this implementation is a binary RBM trained using Contrastive Divergence (CD-k). This RBM serves as an effective feature extractor. Following the unsupervised training of the RBM, its weights are frozen, and the activations of its hidden layer are used as input features for a subsequent classification network.

The classifier is a single-layer neural network that outputs a 4-bit Gray code. This output is then compared against a predefined set of Gray codes for the digits 0-9 to determine the final classification, using the L2 distance for the closest match.
How to Run

    Prerequisites:
        Python 3.x
        PyTorch
        TorchVision
        Matplotlib
        NumPy

    Execution:
    Run the script from your terminal:
    Bash

    python prbm2.py

    The script will automatically download the MNIST dataset, train the RBM, and then train the classifier. Progress, including reconstruction loss for the RBM and accuracy for the classifier, will be printed to the console. Reconstructions from the RBM will be saved as PNG images at regular intervals during training (e.g., rbm_reconstructions_epoch_5.png).

Code Structure

    Configuration: Global hyperparameters for the RBM and the classifier, such as the number of hidden units, learning rates, batch sizes, and epochs.
    Data Loading: Handles the downloading, transformation (flattening and binarizing), and loading of the MNIST dataset for both training and testing phases.
    RBM Implementation:
        RBM class: Defines the RBM architecture, including weights and biases.
        sample_h_given_v and sample_v_given_h: Functions for Gibbs sampling steps.
        forward: Implements the Contrastive Divergence (CD-k) algorithm.
    RBM Training: The training loop for the RBM, which manually updates the weights and biases based on the CD-k algorithm.
    Classifier Implementation:
        GrayCodeClassifier class: A simple neural network that maps RBM features to a 4-bit output.
        decode: A method to translate the 4-bit output into a digit prediction by finding the closest Gray code.
        gray_code_loss: A custom loss function (Binary Cross-Entropy) to compare the network's output with the target Gray codes.
    Classifier Training & Evaluation: The training loop for the classifier, which uses the frozen RBM as a feature extractor. It evaluates the final model accuracy on the test set.

Hyperparameters

Key hyperparameters can be adjusted at the top of the script:

    RBM:
        N_HIDDEN: Number of hidden neurons in the RBM.
        K_CD: Number of Gibbs sampling steps for Contrastive Divergence.
        RBM_EPOCHS: Number of training epochs for the RBM.
        RBM_LEARNING_RATE: Learning rate for the RBM training.
    Classifier:
        CLASSIFIER_EPOCHS: Number of training epochs for the classifier.
        CLASSIFIER_LEARNING_RATE: Learning rate for the classifier training.
