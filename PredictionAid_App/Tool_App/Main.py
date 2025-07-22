# To run type in terminal:
# cd Tool_App
# python -m streamlit run Main.py or C:\Users\sdv.werkstudent\.conda\envs\tool_app\python.exe -m streamlit run Main.py
import streamlit as st
from streamlit import __main__
from PIL import Image

image1 = "DeVoogt_logo.jpg"
image2 = "Feadship_logo.jpg"
image3 = "DesignAID_logo.png"

st.image(image1, width=800, use_container_width=False)
col1, col2 = st.columns([1, 1])
with col1:
    st.image(image2, use_container_width=True)
with col2:
    st.image(image3, use_container_width=True)

st.title("PredictionAID App")
#st.write(os.getcwd())

st.markdown("""
Use the sidebar to navigate between the different available tools in the PredictionAID App. The App contains three tools with different purposes. These will be explained below:
- **Prediction: OLS Regression (linear)**: This tool allows you to perform Ordinary Least Squares (OLS) regression analysis on your dataset. You can upload your dataset, select dependent and independent variables, and run the regression to predict outcomes. 
- **Prediction: Random Forest AI (nonlinear)**: This tool allows you to perform Random Forest regression analysis on your dataset. Similar to OLS Regression, you can upload your dataset, select dependent and independent variables, and run the regression to predict outcomes using a more complex model that can capture non-linear relationships.
- **Selection: Variable Importances Tool (linear + nonlinear)**: This tool helps you to explore and evaluate the relationships between different variables in your dataset. You can visualize correlations and other statistical properties of your data. The goal is to filter the most important variables for predictive purposes.
- **Correlation Matrix**: This tool provides a visual representation of the correlation between different variables in your dataset. You can filter the matrix to focus on specific variables and understand their relationships better.""")


# TO DO:
# - kubus volume kan je niet aanpassen !
# Correlatie matrix filter automatiseren
# User friendly maken, interpreteerbaar maken metrics: MAPE of confidence intervals, voorspellings range
# navigeren makkelijker maken, onthouden geschiedenis, Variable Tool in subkopje
# Eventueel predictions Variable Tool verbeteren (voor nu weggehaald)
# Inleidend tekstje schrijven