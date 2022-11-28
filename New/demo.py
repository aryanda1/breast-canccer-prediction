import streamlit as st
import numpy as np
import pandas as pd
import main

st.markdown("<h1 style='text-align: center; color: red; text-decoration: underline; '><em>Breast Cancer Prediction</em></h1>", unsafe_allow_html=True)

spectra = st.file_uploader("upload file", type={"csv", "txt"})
if spectra is not None:
    spectra_df = pd.read_csv(spectra)

    st.write(spectra_df)

    st.write("Output = ", main.fn(spectra_df.values))
