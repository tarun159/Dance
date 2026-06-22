# Program to implement Naive Bayes using Olivetti Faces Dataset

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
faces = fetch_olivetti_faces()

X = faces.data
y = faces.target
images = faces.images

# Split data
xtrain, xtest, ytrain, ytest, imgtrain, imgtest = train_test_split(
    X, y, images, test_size=0.2, random_state=42
)

# Train model
model = GaussianNB()
model.fit(xtrain, ytrain)

# Predict
ypred = model.predict(xtest)

# Accuracy
print("Accuracy:", accuracy_score(ytest, ypred) * 100)

# Display predictions
plt.figure(figsize=(10,8))

for i in range(10):
    plt.subplot(5, 5, i+1)
    plt.imshow(imgtest[i], cmap='gray')
    plt.title(f"Predicted:{ypred[i]}\nOriginal:{ytest[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()