'''Provides user friendly interface for better understanding of the working of Fashion Reccommendation System.'''

# import necessary libraries
import os                       # for file operations
import pickle                   # for loading pickled files
import tensorflow               # for building the deep learning model
import pybase64                 # for encoding image in base64 format
import streamlit as st          # for creating the web app
from PIL import Image           # for image processing
import numpy as np              # for numerical operations
from numpy.linalg import norm   # for computing a vector or matrix norm
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

# set the title of the app
st.title('Fashion Recommendation System')

# function to save the uploaded file to the 'uploads' folder
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

# Define a function to extract features from an image using the pre-trained ResNet152 model
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

# function to recommend similar images using k-nearest neighbors
def recommend(features, feature_list):
    # create a NearestNeighbors object with 5 neighbors, brute force algorithm, and Euclidean distance metric
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    # fit the NearestNeighbors object to a dataset of feature vectors called feature_list
    neighbors.fit(feature_list)
    # find indices of the 5 nearest neighbors of a given feature vector called normalized_result using the NearestNeighbors object
    distances, indices = neighbors.kneighbors([features])
    # return the indices of the 5 nearest neighbors
    return indices

# function to encode a binary file in base64 format
def get_pybase64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()

    return pybase64.b64encode(data).decode()

# function to set the background image of the app
def set_background(png_file, opacity, blur):
    bin_str = get_pybase64(png_file)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        opacity: {opacity};
    }}

    .stApp:before {{
        content: "";
        position: absolute;
        background-size: cover;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-image: inherit;
        filter: blur({blur});
        z-index: -1;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# set the background image of the app
background_image = 'background.png'
set_background(background_image, 0.9, '2px')

# main app code
uploaded_file = st.file_uploader("Choose an image")

# check if a file has been uploaded
if uploaded_file is not None:
    # try to save the uploaded file
    if save_uploaded_file(uploaded_file):
        # display the uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image)

        # extract features from the uploaded image using a pre-trained model
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

        # get similar images based on the extracted features
        indices = recommend(features, feature_list)

        # display the similar images in a row of 5 columns
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        # display an error message if the file couldn't be saved
        st.header("Some error occurred in file upload")