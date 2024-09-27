import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

st.header('VTS Image Classification')

# Load the pre-trained model
model = load_model(r'model.keras')

# Define the categories
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
            'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno',
            'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
            'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato',
            'turnip', 'watermelon']

# Update image dimensions to match the input size of the model (150x150)
img_height = 150
img_width = 150

# Move the file uploader to the sidebar
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Add a submit button
if st.sidebar.button('Submit'):
    if uploaded_file is not None:
        # Load and preprocess the image
        image = Image.open(uploaded_file)
        image = image.resize((img_width, img_height))  # Resize to 150x150
        img_arr = tf.keras.preprocessing.image.img_to_array(image) / 255.0  # Normalize
        img_bat = tf.expand_dims(img_arr, axis=0)  # Create a batch axis

        # Predict the class of the image
        predict = model.predict(img_bat)
        score = tf.nn.softmax(predict[0])  # Apply softmax to get probabilities

        # Display the results
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write('Veg/Fruit in image is ' + data_cat[np.argmax(score)])
        st.write('With accuracy of {:0.2f}%'.format(np.max(score) * 100))
    else:
        st.write("Please upload an image file.")
