import streamlit as st
import pandas as pd
import pickle

# Page settings
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load model
with open("fraud_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("💳 Credit Card Fraud Detection System")
st.write("Upload a CSV file containing transaction data and detect fraudulent transactions.")

# File upload
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        # Read CSV
        data = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data")
        st.dataframe(data.head())

        # Remove target column if present
        if "Class" in data.columns:
            data = data.drop("Class", axis=1)

        # Expected columns
        expected_columns = [
            'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
            'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13',
            'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
            'V20', 'V21', 'V22', 'V23', 'V24', 'V25',
            'V26', 'V27', 'V28', 'Amount'
        ]

        # Check columns
        missing_cols = [
            col for col in expected_columns
            if col not in data.columns
        ]

        if missing_cols:
            st.error(
                f"Missing columns: {missing_cols}"
            )

        else:

            # Arrange columns correctly
            data = data[expected_columns]

            # Prediction
            predictions = model.predict(data)

            # Add prediction column
            result = data.copy()
            result["Prediction"] = predictions

            st.subheader("✅ Prediction Results")
            st.dataframe(result)

            fraud_count = sum(predictions)
            genuine_count = len(predictions) - fraud_count

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Fraud Transactions",
                    fraud_count
                )

            with col2:
                st.metric(
                    "Legitimate Transactions",
                    genuine_count
                )

            # Download results
            csv = result.to_csv(index=False)

            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="fraud_detection_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("📤 Please upload a CSV file.")