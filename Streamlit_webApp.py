import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Constants
IMG_SIZE = (224, 224)
CLASSES = ['1st Degree Burn', '2nd Degree Burn', '3rd Degree Burn']

# Load model
@st.cache(allow_output_mutation=True)
def load_my_model():
    model = tf.keras.models.load_model('my_model.h5')
    return model

# Function to preprocess image
def preprocess_image(img):
    # Convert the PIL image to a numpy array
    img_array = np.array(img)

    # Load the image with the correct shape for the model
    img = Image.fromarray(img_array)
    img = img.resize(IMG_SIZE)
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255

    return img

# Define main function
def main():
    st.title("Skin Burn Detection")

    # Create file uploader widget
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    # Check if image is uploaded
    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Preprocess image
        img = preprocess_image(Image.open(uploaded_file))

        # Load model
        model = load_my_model()

        # Get prediction
        prediction = model.predict(img)

        # Get class name
        class_name = CLASSES[np.argmax(prediction)]

        # Display result
        st.write("Prediction: ", class_name)

# Run the app
if __name__ == "__main__":
    main()
