from cProfile import label
import streamlit as st
from helper import fetch_options, fetch_data, train_model

st.set_page_config(
     page_title="Crypto Price Prediction",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://github.com/everydaycodings/Anima-Recommendation-System-WebApp#readme',
         'Report a bug': "https://github.com/everydaycodings/Anima-Recommendation-System-WebApp/issues/new/choose",
         'About': "# This is a header. This is an *extremely* cool app!"
     }
)

st.title("Crypto Price Prediction")

selected_asset_name = st.selectbox("Select or Enter The Name of the Crypto Asset", options = fetch_options())
st.number_input(label="Enter for how many future days you want to predict the {} market".format(selected_asset_name), min_value=10, max_value=100, value=30)

if st.button("Apply"):
    data  = fetch_data(selected_asset_name)
    model = train_model(data)
    #print(model.history["val_loss"][-1])
