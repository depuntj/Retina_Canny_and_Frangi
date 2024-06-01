#Link Dataset Tomat : https://www.kaggle.com/datasets/nexuswho/laboro-tomato

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

# Paths to your folders
ripe_folder = 'Matang'
unripe_folder = 'Mentah'
half_ripe_folder = 'Setengah Matang'

# Function to preprocess images and extract features using OpenCV
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))  # Resize to a fixed size

    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # # Unripe Mask
    # lower_unripe = np.array([14, 100, 100], dtype=np.uint8)
    # upper_unripe = np.array([47, 255, 255], dtype=np.uint8)
    # mask_unripe = cv2.inRange(hsv_image, lower_unripe, upper_unripe)
    #
    # # Half Ripe Mask
    # lower_half_ripe = np.array([0, 100, 100], dtype=np.uint8)
    # upper_half_ripe = np.array([10, 255, 255], dtype=np.uint8)
    # mask_half_ripe = cv2.inRange(hsv_image, lower_half_ripe, upper_half_ripe)
    #
    # # Ripe Mask
    # lower_ripe = np.array([170, 100, 100], dtype=np.uint8)
    # upper_ripe = np.array([180, 255, 255], dtype=np.uint8)
    # mask_ripe = cv2.inRange(hsv_image, lower_ripe, upper_ripe)

    # Unripe Mask
    lower_unripe = np.array([14, 100, 100], dtype=np.uint8)
    upper_unripe = np.array([47, 255, 255], dtype=np.uint8)
    mask_unripe = cv2.inRange(hsv_image, lower_unripe, upper_unripe)
    cv2.imwrite("images/1_unripe.png", mask_unripe)

    # Ripe Mask
    # lower_half_ripe = np.array([0, 100, 100], dtype=np.uint8)
    # upper_half_ripe = np.array([10, 255, 255], dtype=np.uint8)
    lower_half_ripe = np.array([0, 100, 100], dtype=np.uint8)
    upper_half_ripe = np.array([10, 255, 255], dtype=np.uint8)
    mask_ripe = cv2.inRange(hsv_image, lower_half_ripe, upper_half_ripe)
    cv2.imwrite("images/1_ripe.png", mask_ripe)

    # Half Ripe Mask
    # lower_ripe = np.array([25, 100, 100], dtype=np.uint8)
    # upper_ripe = np.array([35, 255, 255], dtype=np.uint8)
    lower_ripe = np.array([10, 100, 100], dtype=np.uint8)
    upper_ripe = np.array([14, 255, 255], dtype=np.uint8)
    mask_half_ripe = cv2.inRange(hsv_image, lower_ripe, upper_ripe)
    cv2.imwrite("images/1_halfripe.png", mask_half_ripe)

    # Calculate the mean color within the masked areas
    mean_ripe = cv2.mean(hsv_image, mask=mask_ripe)[:3]
    mean_unripe = cv2.mean(hsv_image, mask=mask_unripe)[:3]
    mean_half_ripe = cv2.mean(hsv_image, mask=mask_half_ripe)[:3]

    # Combine the mean colors into a feature vector
    features = np.concatenate([mean_ripe, mean_unripe, mean_half_ripe])

    return features


# Lists to hold features and labels
features = []
labels = []

# Process ripe images
for filename in os.listdir(ripe_folder):
    if filename.endswith('.jpg'):
        features.append(preprocess_image(os.path.join(ripe_folder, filename)))
        labels.append(1)  # Label for ripe

# Process unripe images
for filename in os.listdir(unripe_folder):
    if filename.endswith('.jpg'):
        features.append(preprocess_image(os.path.join(unripe_folder, filename)))
        labels.append(0)  # Label for unripe

# Process half-ripe images
for filename in os.listdir(half_ripe_folder):
    if filename.endswith('.jpg'):
        features.append(preprocess_image(os.path.join(half_ripe_folder, filename)))
        labels.append(2)  # Label for half-ripe

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create and train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)  # You can choose the number of neighbors
knn.fit(X_train, y_train)

# Save the model to a file
model_filename = 'knn_model.joblib'
dump(knn, model_filename)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
