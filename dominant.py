import streamlit as st

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

st.title("Dominant Color Extraction")

st.subheader('Input Image')
img = st.file_uploader('Choose an image')

if img is not None:
    st.header('Original Image')
    st.image(img)


    # KMEANS CODE
    img = plt.imread(img)

    n = img.shape[0]*img.shape[1]
    all_pixels = img.reshape((n, 3))

    model  = KMeans(n_clusters = 8)
    model.fit(all_pixels)

    centers = model.cluster_centers_.astype('uint8')

    new_img = np.zeros((n, 3), dtype='uint8')

    for i in range(n):
        group_idx = model.labels_[i]
        new_img[i] = centers[group_idx]

    new_img = new_img.reshape(*img.shape)

    st.header('Modified Image')
    st.image(new_img)
