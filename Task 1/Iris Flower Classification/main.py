import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load your own dataset (replace 'your_dataset.csv' with your actual file path)
data = pd.read_csv('C:\\Users\\Mehran\\PycharmProjects\\Iris Flower Classification\\IRIS.csv')

X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Assuming 'y' is the column containing class labels like 'Iris-setosa', 'Iris-versicolor', etc.
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Now 'y_encoded' contains numerical labels instead of string labels
plt.scatter(X['sepal_length'], X['sepal_width'], c=y_encoded, cmap='viridis')
plt.show()

# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Evaluate the model on the test set
accuracy = knn.score(X_test, y_test)
print(f'Model Accuracy: {accuracy:.2f}')

# Example new data
new_data = [[5.1, 3.5, 1.4, 0.2], [6.0, 3.0, 4.8, 1.8]]

# Use the trained model to make predictions
predictions = knn.predict(new_data)

# Display the predicted species
print(f'Predicted Species: {predictions}')