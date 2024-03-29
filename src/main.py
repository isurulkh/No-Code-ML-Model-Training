import os
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from ml_utility import (read_data, preprocess_data, train_model, evaluate_model, parent_dir)
import pandas as pd

# Set page config
st.set_page_config(page_title="Automate ML", page_icon="🧠", layout="wide")

# Sidebar for configuration
with st.sidebar:
    st.title("🤖 No Code ML Model Training")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel)", type=['csv', 'xlsx', 'xls'])

    # Option to select from preloaded datasets if no file is uploaded
    if uploaded_file is None:
        working_dir = os.getcwd()
        dataset_list = os.listdir(os.path.join(working_dir, "data"))
        dataset = st.selectbox("Select a preloaded dataset", dataset_list, index=0)
    else:
        dataset = uploaded_file

# Main page
st.header("Dataset and Model Configuration")

# Data Loading with Progress Bar
with st.spinner('Loading data...'):
    df = read_data(dataset)

if df is not None:
    # Data Upload and Preview Section
    st.subheader("Data Upload and Preview")
    with st.expander("Show/Hide Dataset"):
        selected_columns = st.multiselect("Select columns to display", df.columns)
        st.dataframe(df[selected_columns].head())

     # Data Preprocessing Section (example)
    st.subheader("Data Preprocessing (Optional)")
    with st.expander("Preprocessing Options"):
        # Numerical Feature Scaling
        st.checkbox("Scale Numerical Features", key="scale_numerical")
        if st.session_state.scale_numerical:
            numerical_cols = df.select_dtypes(include=['number']).columns
            selected_num_cols = st.multiselect("Select numerical columns to scale", numerical_cols)

        # Categorical Feature Encoding (example)
        st.checkbox("Encode Categorical Features", key="encode_categorical")
        if st.session_state.encode_categorical:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            selected_cat_cols = st.multiselect("Select categorical columns to encode", categorical_cols)
            
            
    # Model Training Section
    st.subheader("Model Training")
    col1, col2, col3, col4 = st.columns(4)

    scaler_type_list = ["StandardScaler", "MinMaxScaler"]

    model_dictionary = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Classifier (SVC)": SVC(),
        "Random Forest Classifier": RandomForestClassifier(),
        "XGBoost Classifier": XGBClassifier()
    }

    with col1:
        target_column = st.selectbox("Select the Target Column", list(df.columns))
    with col2:
        scaler_type = st.selectbox("Select a scaler", scaler_type_list)
    with col3:
        selected_model = st.selectbox("Select a Model", list(model_dictionary.keys()))
    with col4:
        model_name = st.text_input("Model name", value="MyModel")

    if st.button("Train the Model"):
        with st.spinner('Training the model...'):
            # Preprocessing with Progress Bar (example)
            with st.empty():  # Placeholder for progress bar
                X_train, X_test, y_train, y_test = preprocess_data(df, target_column, scaler_type)
                st.progress(100)  # Show 100% progress after preprocessing

            model_to_be_trained = model_dictionary[selected_model]

            model = train_model(X_train, y_train, model_to_be_trained, model_name)

            accuracy, other_metrics = evaluate_model(model, X_test, y_test)
            st.success(f"Model trained successfully! Test Accuracy: {accuracy}")

            # Display additional evaluation metrics
            with st.expander("Show Detailed Evaluation Metrics"):
                st.write(other_metrics)

            # Download button after successful training
            st.download_button(
                label="Download Trained Model",
                data=open(f"{parent_dir}/trained_model/{model_name}.pkl", "rb").read(),
                file_name=f"{model_name}.pkl",
                mime="application/octet-stream"
            )

else:
    st.warning("Please select or upload a dataset to proceed.")