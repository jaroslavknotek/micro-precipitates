import streamlit as st
import precipitates.nn as nn
import precipitates.visualization as visualization
import precipitates.precipitate as precipitate
import pandas as pd
import numpy as np
from PIL import Image
import logging


ACCEPTED_TYPES =[
    'png', 
    'jpg',
    'jpeg',
    'tif',
    'tiff'
]

model_path = "model/model-20230417-morp_opening.h5"


def predict(model_path,img):
    img_cropped = img[:img.shape[1]]
    img = img_cropped.astype(float)/np.max(img_cropped)
    model = load_model(model_path)
    return nn.predict(model,img_cropped)

def load_model(model_path):
    model = nn.compose_unet((128,128))
    model.load_weights(model_path)
    return model

def _get_feature_dataset(shapes):
    features = [precipitate.extract_features(shape) for shape in shapes]
    shape_classes = [ precipitate.classify_shape(feature) for feature in features]
    
    df = pd.DataFrame(features)
    df['shape_class'] = shape_classes
    return df

def _process_image(img,pred,px2um=None):
    contoured =visualization.add_contours_morph(img,pred,contour_width=2)
   
    shapes = precipitate.identify_precipitates_from_mask(pred)
    df_features = _get_feature_dataset(shapes)

    if px2um is not None:
        _add_micrometer_scale(df_features,px2um)
    else:
        logging.warning(f"No scale given")

    fig_hist = visualization.plot_histograms(df_features)
    fig_details = visualization.plot_precipitate_details(df_features,pred,img)
    return {
        "pred":pred,
        "contoured":contoured,
        "df":df_features,
        "fig_hist":fig_hist,
        "fig_details":fig_details
    }

uploaded_file = st.file_uploader("Upload Image",type=ACCEPTED_TYPES)

if uploaded_file is not None: 
    st.write("Input Image")
    u_img = Image.open(uploaded_file,formats=None)
    img = np.array(u_img) 
    img_f = img.astype(float) / np.max(img) 
    show = st.image(img_f)
   
    with st.spinner('Segmentation in progress ...'):            
        prediction = predict(model_path,img_f)
        st.write("Segmentation Mask")
        st.image(prediction,clamp=True)
       
    with st.spinner("Processin Mask"):
        res = _process_image(img_f,prediction)

    with st.spinner("Drawing results"):
        st.write("Contours:")
        st.image(res['contoured'],clamp=True)
        st.write("Distribution")
        st.download_button(
              label="Download data as CSV",
              data=res['df'].to_csv().encode('utf-8'),
              file_name='distribution.csv',
              mime='text/csv',
        )
        st.write(res['df'])
        st.write("Histogram")
        st.pyplot(res['fig_hist'])
        st.write("Details")
        st.pyplot(res['fig_details'])

        st.success('Done')
