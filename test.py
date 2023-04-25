'''For testing purpose of the system. Returns the indices of the 5 nearest neighbors.'''

# import necessary libraries
import pickle                   # for loading pickled file
import tensorflow               # for building the deep learning model
import numpy as np              # for numerical operations
from numpy.linalg import norm   # for computing a vector or matrix norm
import cv2                      # for image and video processing
# for loading and preprocessing images to be used in deep learning models
from tensorflow.keras.preprocessing import image
# for performing max pooling operation on the spatial dimensions of 2D inputs
from tensorflow.keras.layers import GlobalMaxPooling2D
# for use in transfer learning applications
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
# provides unsupervised machine learning methods for nearest neighbor search and clusterin
from sklearn.neighbors import NearestNeighbors

# load feature embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load the pre-trained ResNet50 model with pre-trained weights
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze the layers of the pre-trained model to prevent them from being updated during training
model.trainable = False

# Create a new Keras Sequential model by combining the pre-trained ResNet50 model with a global max pooling layer
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Load the image from the given path and resize it to the input shape of the pre-trained model
img = image.load_img('uploads\saree.jpg', target_size=(224, 224))
# Convert the image to a NumPy array
img_array = image.img_to_array(img)
# Add a new dimension to the array to create a batch of size 1
expanded_img_array = np.expand_dims(img_array, axis=0)
# Preprocess the image by applying the same preprocessing steps used during training of the pre-trained model
preprocessed_img = preprocess_input(expanded_img_array)
# Use the pre-trained model to extract features from the preprocessed image
result = model.predict(preprocessed_img).flatten()
# Normalize the feature vector to have unit length
normalized_result = result / norm(result)

# create a NearestNeighbors object with 5 neighbors, brute force algorithm, and Euclidean distance metric
neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')

# fit the NearestNeighbors object to a dataset of feature vectors called feature_list
neighbors.fit(feature_list)

# find indices of the 5 nearest neighbors of a given feature vector called normalized_result using the NearestNeighbors object
distances, indices = neighbors.kneighbors([normalized_result])

# print the indices of the 5 nearest neighbors
print(indices) 

# loop through the indices of the 5 nearest neighbors, excluding the first index which corresponds to the input feature vector
for file in indices[0]:
    # load the image corresponding to the current index using OpenCV's imread function and the filenames list
    temp_img = cv2.imread(filenames[file])
    # resize the image to 512x512 pixels and display it using OpenCV's imshow function
    cv2.imshow('output', cv2.resize(temp_img, (512, 512)))
    # wait for a key event before closing the image window, this allows the user to view the image before moving on to the next one
    cv2.waitKey(0)