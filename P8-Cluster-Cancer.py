# 1. IMPORTING THE LIBRARIES AND LOADING THE DATASET
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv('Datasets\\P8-CancerData.csv')

# ======================================================================================================================


# 2. SELECTING THE FEATURES AND GENERATING THE FEATURE VECTORS.
feat_cols_sm = ["radius_mean", "concavity_mean", "symmetry_mean"]
features = np.array(data[feat_cols_sm])

# ======================================================================================================================


# 3. IMPLEMENTING THE KMeans CLUSTERING MODEL.
# Initialize the KMeans cluster module.
# Setting it to find two clusters, hoping to find malignant vs benign.
clusters = KMeans(n_clusters=2, n_init=10, max_iter=300)

# Fit model to our selected features.
clusters.fit(features)

# Put centroids and results into variables.
centroids = clusters.cluster_centers_
labels = clusters.labels_

# Sanity check
print("Sanity check: Centroids")
print(centroids)

# ======================================================================================================================


# 4. VISUALIZING THE CLUSTERS.

fig = plt.figure()  # Create new MatPlotLib figure
ax = fig.add_subplot(111, projection='3d')  # Add 3rd dimension to figure
colors = ["r", "b"]  # This means "red" and "blue"

# Plot all the features and assign color based on cluster identity label
for i in range(len(features)):
    ax.scatter(xs=features[i][0], ys=features[i][1], zs=features[i][2],
               c=colors[labels[i]], zdir='z')

# Plot centroids, though you can't really see them.
ax.scatter(xs=centroids[:, 0], ys=centroids[:, 1], zs=centroids[:, 2],
           marker="x", s=150, c="c")

# Create array of diagnosis data, which should be same length as labels.
diag = np.array(data['diagnosis'])
# Create variable to hold matches in order to get percentage accuracy.
matches = 0

# Transform diagnosis vector from B||M to 0||1 and matches++ if correct.
for i in range(0, len(diag)):
    if diag[i] == "B":
        diag[i] = 0
    if diag[i] == "M":
        diag[i] = 1
    if diag[i] == labels[i]:
        matches = matches + 1

# Calculate percentage matches and print.
percentMatch = (matches / len(diag)) * 100
print("Percent matched between benign and malignant ", percentMatch)

# Set labels on figure and show 3D scatter plot to visualize data and clusters.
ax.set_xlabel("Radius Mean")
ax.set_ylabel("Concavity Mean")
ax.set_zlabel("Symmetry Mean")
plt.show()
