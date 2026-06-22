# Step 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Step 2: Load the Iris dataset
iris = load_iris()
data = iris.data
labels = iris.target
label_names = iris.target_names

# Step 3: Convert to DataFrame (optional, for understanding)
iris_df = pd.DataFrame(data, columns=iris.feature_names)
print("Dataset Preview:\n", iris_df.head())

# Step 4: Apply PCA to reduce dimensions from 4 to 2
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data)

# Step 5: Create New DataFrame for reduced data
reduced_df = pd.DataFrame(data_reduced, columns=['Principal Component 1', 'Principal Component 2'])
reduced_df['Label'] = labels

# Step 6: Plot the PCA result
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']

for i, label in enumerate(np.unique(labels)):
    plt.scatter(
        reduced_df[reduced_df['Label'] == label]['Principal Component 1'],
        reduced_df[reduced_df['Label'] == label]['Principal Component 2'],
        label=label_names[label],
        color=colors[i]
    )

# Step 7: Add labels and show plot
plt.title('PCA on Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()
