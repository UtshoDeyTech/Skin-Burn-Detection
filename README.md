# Skin Burn Detection

The Skin Burn Detection web app is a deep learning model that predicts the degree of a skin burn based on an uploaded image. The model is trained using the VGG19 architecture and the ImageNet dataset, which is a large-scale dataset of annotated images used for training deep learning models. The app is built using Streamlit, an open-source Python library that allows developers to create interactive web applications with ease.

To use the web app, the user needs to follow the instructions provided in the repository. They need to download the repository and install the required libraries listed in the `requirements.txt` file. Once the libraries are installed, the user can launch the web app by running the `streamlit_webApp.py` file. This will open a local web server and launch the app in the user's web browser. The user can then upload an image of a burn, and the app will predict the degree of the burn based on the trained model.

Overall, the Skin Burn Detection web app is a useful tool for quickly assessing the severity of a skin burn and determining the appropriate course of treatment.

## Installation

1. Download the code from the [GitHub repository](https://github.com/UtshoDeyTech/Skin-Burn-Detection).

   ```
   git clone https://github.com/UtshoDeyTech/Skin-Burn-Detection
   ```

2. Create a virtual environment in Python using the following command:

   ```
   python -m venv env
   ```

3. Activate the virtual environment:

   **Windows**

   ```
   env\Scripts\activate
   ```

   **Mac/Linux**

   ```
   source env/bin/activate
   ```

4. Install the required libraries using the following command:

   ```
   pip install -r requirements.txt
   ```

5. Launching the web app : To launch the web app, the user needs to run the `streamlit_webApp.py` file. This will open a local web server and launch the app in the user's web browser. The user can then upload an image of a burn and the app will predict the degree of the burn based on the trained model. For lunching the app, type in the terminal-
   ```
   streamlit run streamlit_webApp.py
   ```

## Usage

To run the skin burn detection web app, run the following command:

This will launch the web app in your browser. Upload an image of a burn and the app will predict the degree of the burn.

## Demo Images

Here are some demo images that you can use to test the skin burn detection web app:

<div style="display:flex;">
    <img src="https://raw.githubusercontent.com/UtshoDeyTech/Skin-Burn-Detection/master/1st.PNG" alt="First Degree Burn" width="400" height="400"/> 
    <img src="https://raw.githubusercontent.com/UtshoDeyTech/Skin-Burn-Detection/master/2nd.PNG" alt="Second Degree Burn" width="400" height="400"/> 
    <img src="https://raw.githubusercontent.com/UtshoDeyTech/Skin-Burn-Detection/master/3rd.PNG" alt="Third Degree Burn" width="400" height="400"/>
</div>

## Code Explanation

- `skin burn.py` file: This file contains the code for training the deep learning model using the VGG19 architecture and the ImageNet dataset. The VGG19 architecture is a widely used deep learning model for image classification tasks, and the ImageNet dataset is a large-scale dataset of annotated images used for training deep learning models. The skin_burn.py file loads the ImageNet dataset and fine-tunes the VGG19 model on the skin burn classification task. After training, the model is saved as `my_model.h5` for later use by the web app.

- `streamlit_webApp.py` file: This file contains the code for the web app using Streamlit. Streamlit is an open-source Python library that allows developers to create interactive web applications with ease. The `streamlit_webApp.py` file loads the trained model from `my_model.h5` and uses it to predict the degree of the burn in the uploaded image. When the user uploads an image of a burn, the app displays the uploaded image and predicts the degree of the burn based on the output of the model. The predicted degree of the burn is displayed as one of three categories: first-degree burn, second-degree burn, or third-degree burn.

- `data` folder: This folder contains the dataset used for training the model. The dataset consists of annotated images of burns with corresponding labels indicating the degree of the burn. The dataset is used to train the deep learning model in `skin_burn.py`.

- `requirements.txt` file: This file contains the required libraries to run the code. The libraries include TensorFlow, Keras, Streamlit, and other dependencies needed to train the deep learning model and run the web app.
