'''Feature extraction of every procuct's image present in the  database and dumping them as '.pkl' files, i.e. "filenames.pkl" and "embeddings.pkl".'''

# import necessary libraries
import os                       # for file operations
import pickle                   # for loading and dumping pickled file
import tensorflow               # for building the deep learning model
import numpy as np              # for numerical operations
from numpy.linalg import norm   # for computing a vector or matrix norm
from tqdm import tqdm  # for displaying progress bars in a loop
# for loading and preprocessing images to be used in deep learning models
from tensorflow.keras.preprocessing import image
# for performing max pooling operation on the spatial dimensions of 2D inputs
from tensorflow.keras.layers import GlobalMaxPooling2D
# for use in transfer learning applications
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input

# Load the pre-trained ResNet50 model with pre-trained weights
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze the layers of the pre-trained model to prevent them from being updated during training
model.trainable = False

# Create a new Keras Sequential model by combining the pre-trained ResNet50 model with a global max pooling layer
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Print a summary of the new model to check its architecture
print(model.summary())

# Define a function to extract features from an image using the pre-trained ResNet50 model
def feature_extraction(img_path, model):
    try:
        # Load the image from the given path and resize it to the input shape of the pre-trained model
        img = image.load_img(img_path, target_size=(224, 224))
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
        # Returning the result
        return normalized_result
    except:
        # If there is any error while processing the image, print an error message and return None
        print(f"Error processing file: {img_path}")
        return None

# Get a list of all image file paths in the "images" directory
filenames = []
for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))

# Extract features from each image file using the pre-trained model and store them in a list
feature_list = []
for file in tqdm(filenames):
    feature_list.append(feature_extraction(file, model))

# Save the list of feature vectors and image file names to disk as pickle files
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))