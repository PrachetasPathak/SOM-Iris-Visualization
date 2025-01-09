# SOM-Iris-Visualization
This project demonstrates the use of a Self-Organizing Map (SOM) to visualize and analyze the Iris dataset. The SOM is trained on standardized features of the dataset, and its output is visualized as a grid displaying winning neurons and the U-Matrix.
Title
Self-Organizing Map (SOM): Iris Dataset

Overview
This project implements a Self-Organizing Map (SOM) using the minisom library to cluster and visualize the Iris dataset. The SOM grid and U-Matrix provide insights into the relationships between data points and the topological structure of the dataset.

Features
Clustering and visualization of the Iris dataset using a 10x10 SOM grid.
Representation of neuron activations for different classes of the Iris dataset.
Visualization of the U-Matrix to analyze the distance between neighboring neurons.
Dataset
Source: The Iris dataset is loaded from the sklearn.datasets module.
Features Used: All four features:
Sepal Length
Sepal Width
Petal Length
Petal Width
Target Classes: Three species of Iris flowers:
Setosa
Versicolor
Virginica
Code Workflow
Data Preparation:

Load the Iris dataset and standardize its features using StandardScaler.
Optionally shuffle the dataset to ensure a randomized training process.
SOM Initialization:

Define a 10x10 SOM grid (MiniSom) with:
A sigma value of 1.0 to control the neighborhood radius.
A learning rate of 0.5 for adaptive weight adjustments.
Training:

Train the SOM using 100 random iterations on the standardized dataset.
Visualization:

SOM Grid: Shows the neurons activated by the input data points, color-coded by class.
U-Matrix: Displays the average distance between neurons, indicating cluster boundaries.
Legend: Provides class information using distinct colors for:

Red: Setosa
Green: Versicolor
Blue: Virginica
