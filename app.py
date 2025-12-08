import time
import faiss
import pathlib
from PIL import Image
import numpy as np
import pandas as pd
import os

import streamlit as st
from streamlit_cropper import st_cropper


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

st.set_page_config(layout="wide")

FILES_PATH = str(pathlib.Path().resolve())

# Base path used to resolve image paths stored in the metadata CSV.
IMAGES_BASE_PATH = FILES_PATH

# Path in which the database should be located
DB_PATH = os.path.join(FILES_PATH, 'database')
FAISS_PATH = os.path.join(FILES_PATH, 'faiss_indexes')

DB_FILE = 'db_train.csv' # training metadata for retrieval
@st.cache_data
def load_metadata():
    return pd.read_csv(os.path.join(DB_PATH, DB_FILE))

@st.cache_resource
def load_faiss_index(index_path):
    return faiss.read_index(index_path)

@st.cache_resource
def get_color_histogram_extractor():
    from src.extractors.color_histogram import extract_color_histogram

    return extract_color_histogram

@st.cache_resource
def get_sift_bow_extractor():
    from src.extractors.sift_bow import extract_sift_bow

    return extract_sift_bow

@st.cache_resource
def get_orb_bow_extractor():
    from src.extractors.orb_bow import extract_orb_bow

    return extract_orb_bow

@st.cache_resource
def get_resnet50_extractor():
    from src.extractors.resnet import extract_resnet50

    return extract_resnet50

@st.cache_resource
def get_efficientnet_extractor():
    from src.extractors.efficientnet import extract_efficientnet_b0

    return extract_efficientnet_b0

def retrieve_image(img_query, feature_extractor, n_imgs=11):
    if feature_extractor == 'Color histogram':
        model_feature_extractor = get_color_histogram_extractor()
        index_filename = 'color_histogram_db_train_l2.index'
    elif feature_extractor == 'SIFT BoW':
        model_feature_extractor = get_sift_bow_extractor()
        index_filename = 'sift_bow_db_train_l2.index'
    elif feature_extractor == 'ORB BoW':
        model_feature_extractor = get_orb_bow_extractor()
        index_filename = 'orb_bow_db_train_l2.index'
    elif feature_extractor == 'ResNet50':
        model_feature_extractor = get_resnet50_extractor()
        index_filename = 'resnet50_db_train_l2.index'
    elif feature_extractor == 'EfficientNet-B0':
        model_feature_extractor = get_efficientnet_extractor()
        index_filename = 'efficientnet_b0_db_train_l2.index'
    else:
        raise ValueError(f"Unsupported feature extractor: {feature_extractor}")

    index_path = os.path.join(FAISS_PATH, index_filename)
    indexer = load_faiss_index(index_path)

    embeddings = model_feature_extractor(img_query)
    vector = np.float32(embeddings)
    # No normalization is required for the L2 metric we are using.
    _, indices = indexer.search(vector, k=n_imgs)

    return indices[0]

def main():
    st.title('CBIR IMAGE SEARCH')
    
    col1, col2 = st.columns(2)

    with col1:
        st.header('QUERY')

        st.subheader('Choose feature extractor')
        option = st.selectbox('.', ('Color histogram', 'SIFT BoW', 'ORB BoW', 'ResNet50', 'EfficientNet-B0'))

        st.subheader('Upload image')
        img_file = st.file_uploader(label='.', type=['png', 'jpg'])

        if img_file:
            img = Image.open(img_file)
            # Get a cropped image from the frontend
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0004')
            
            # Manipulate cropped image at will
            st.write("Preview")
            _ = cropped_img.thumbnail((150,150))
            st.image(cropped_img)

    with col2:
        st.header('RESULT')
        if img_file:
            st.markdown('**Retrieving .......**')
            start = time.time()

            retriev = retrieve_image(cropped_img, option, n_imgs=11)
            metadata = load_metadata()
            image_list = list(metadata.image.values)

            end = time.time()
            st.markdown('**Finish in ' + str(end - start) + ' seconds**')

            col3, col4 = st.columns(2)

            with col3:
                image = Image.open(os.path.join(IMAGES_BASE_PATH, image_list[retriev[0]]))
                st.image(image, use_column_width = 'always')

            with col4:
                image = Image.open(os.path.join(IMAGES_BASE_PATH, image_list[retriev[1]]))
                st.image(image, use_column_width = 'always')

            col5, col6, col7 = st.columns(3)

            with col5:
                for u in range(2, 11, 3):
                    image = Image.open(os.path.join(IMAGES_BASE_PATH, image_list[retriev[u]]))
                    st.image(image, use_column_width = 'always')

            with col6:
                for u in range(3, 11, 3):
                    image = Image.open(os.path.join(IMAGES_BASE_PATH, image_list[retriev[u]]))
                    st.image(image, use_column_width = 'always')

            with col7:
                for u in range(4, 11, 3):
                    image = Image.open(os.path.join(IMAGES_BASE_PATH, image_list[retriev[u]]))
                    st.image(image, use_column_width = 'always')

if __name__ == '__main__':
    main()
