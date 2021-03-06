from cProfile import label
import streamlit as st
from helper import fetch_options, fetch_data, train_model, display_loss_plot, display_accuracy_graph_plot, predict

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
     page_title="Crypto Price Prediction",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://github.com/everydaycodings/Crypto-Price-Prediction-using-RNN',
         'Report a bug': "https://github.com/everydaycodings/Crypto-Price-Prediction-using-RNN/issues/new",
         'About': "# This is a header. This is an *extremely* cool app!"
     }
)

st.title("Crypto Price Prediction")


selected_asset_name = st.selectbox("Select or Enter The Name of the Crypto Asset", options = fetch_options())
days = 30#st.number_input(label="Enter for how many future days you want to predict the {} market".format(selected_asset_name), min_value=10, max_value=100, value=30)

if st.button("Predict"):
    data  = fetch_data(selected_asset_name)
    st.dataframe(data)
    st.info("Model will be train within {} second(also depends upon your pc computing power) so please have patience (model epochs=100)".format((100*2) + 10))
    model = train_model(data)
    st.success('Your Model Has been Trainned.')
    col1, col2= st.columns(2)

    with col1:
        st.subheader("Performance Graph For The Trainned Model")
        loss_plot = display_loss_plot(model)
        st.line_chart(loss_plot, height=460)

    with col2:
        st.subheader("Trainned Model working on a Test Dataset")
        st.image(display_accuracy_graph_plot())

    col3, col4 = st.columns(2)

    plot, plot1 = predict()
    with col3:
        st.subheader("{} Future Prdiction for next {} days".format(selected_asset_name, days))
        st.image(plot)
    
    with col4:
        st.subheader("Smoothen {} Future Prdiction for next {} days".format(selected_asset_name, days))
        st.image(plot1)
