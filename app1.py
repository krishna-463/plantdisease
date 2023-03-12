from email.mime import image
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import base64
from tensorflow.keras.utils import load_img, img_to_array



################################################


st.title('PLant Disease Identification')



##################################################


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('bgg.png')

##########################################



st.set_option('deprecation.showfileUploaderEncoding', False)


#@st.cache(allow_output_mutation=True)
def load_model():
    model  =  keras.models.load_model('model/resnet50.h5')
    return model


uploaded_file = st.file_uploader("Choose a file",type = ["jpg" , "png", "jpeg"])

if uploaded_file:


    st.image(uploaded_file)




    size = (224,224)
    uploaded_file = Image.open(uploaded_file)
    image = ImageOps.fit(uploaded_file,size,Image.ANTIALIAS)
    img = np.asarray(image)









    li = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
     'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
     'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
     'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 
     'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 
     'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
     'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
     'Tomato___Spider_mites Two-spotted_spider_mite', 
     'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    
    model = load_model()
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    prediction = model.predict(img)
    d = prediction.flatten()
    j = d.max()
    for index,item in enumerate(d):
        if item == j:
            class_name = li[index]
    st.header(class_name)
else:
    st.header("Upload image to identify")