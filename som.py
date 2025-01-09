import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom
import matplotlib.pyplot as plt

# Load and standardize Iris data
iris = load_iris()
X, y = iris.data, iris.target

'''# Shuffle the dataset
indices = np.arange(X.shape[0])  # Create an array of indices (0 to number of samples - 1)
np.random.shuffle(indices)       # Shuffle the indices
X, y = X[indices], y[indices]    # Reorder X and y according to shuffled indices
'''
# Standardize the feature matrix
X = StandardScaler().fit_transform(X)

# Initialize and train the SOM
som_dim_x, som_dim_y = 10, 10  # Defines a 10x10 grid
som = MiniSom(som_dim_x, som_dim_y, X.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X, 100)

# Plotting the SOM grid and U-Matrix
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
colors = ['r', 'g', 'b']
labels = ['Setosa', 'Versicolor', 'Virginica']

# SOM Grid with data points
for i, x in enumerate(X):
    winner = som.winner(x)
    ax[0].plot(winner[0] + 0.5, winner[1] + 0.5, colors[y[i]] + 'o', markersize=10, alpha=0.7)
ax[0].set_title("SOM Grid")
ax[0].set_xticks(range(som_dim_x + 1))
ax[0].set_yticks(range(som_dim_y + 1))
ax[0].grid(True)


# U-Matrix (mean inter-neuron distance)
u_matrix = som.distance_map()
ax[1].imshow(u_matrix, cmap='bone_r', interpolation='nearest')
ax[1].set_title("U-Matrix")

# Legend
for i, label in enumerate(labels):
    ax[0].plot([], [], colors[i] + 'o', label=label)
ax[0].legend(loc='upper right')
plt.tight_layout()
plt.show()
