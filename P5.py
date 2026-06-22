import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Generate 100 random values
x = np.random.rand(100).reshape(-1,1)

# Label first 50 points
y = np.array([1 if val <= 0.5 else 2 for val in x[:50]])

# Training and testing data
xtrain = x[:50]
ytrain = y
xtest = x[50:]

# Different k values
k_values = [1, 2, 3, 4, 5, 20, 30]

plt.figure(figsize=(12,10))

for i, k in enumerate(k_values, 1):

    # Create and train KNN model
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(xtrain, ytrain)

    # Predict test data
    predicted = model.predict(xtest)

    # Plot results
    plt.subplot(4, 2, i)
    plt.scatter(xtrain, ytrain, color='red', label='Training')
    plt.scatter(xtest, predicted, color='blue', label='Predicted')
    plt.title(f'K = {k}')
    plt.legend()

plt.tight_layout()
plt.show()