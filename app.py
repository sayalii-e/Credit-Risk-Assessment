import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("Credit Risk Assessment Dashboard")
st.write("Enter customer details to predict Approved Flag (P1, P2, P3, P4)")

def user_input_features():
    Age_Newest_TL = st.number_input("Age of Newest TL", min_value=0)
    Age_Oldest_TL = st.number_input("Age of Oldest TL", min_value=0)
    CC_enq_L12m = st.number_input("CC Enquiries Last 12 months", min_value=0)
    CC_Flag = st.selectbox("Has Credit Card?", ['Yes','No'])
    EDUCATION = st.selectbox("Education", ['Graduate','Postgraduate','School','Other'])
    GENDER = st.selectbox("Gender", ['Male','Female'])
    MARITALSTATUS = st.selectbox("Marital Status", ['Single','Married','Divorced','Other'])
    NETMONTHLYINCOME = st.number_input("Net Monthly Income", min_value=0)
    Time_With_Curr_Empr= st.number_input("Time with Current Employer", min_value=0)
    Credit_Score = st.number_input("Credit Score", min_value=0)

    data = {
        'Age_Newest_TL': Age_Newest_TL,
        'Age_Oldest_TL': Age_Oldest_TL,
        'CC_enq_L12m': CC_enq_L12m,
        'CC_Flag': 1 if CC_Flag=='Yes' else 0,
        'EDUCATION': EDUCATION,
        'GENDER': GENDER,
        'MARITALSTATUS': MARITALSTATUS,
        'NETMONTHLYINCOME': NETMONTHLYINCOME,
        'Time_With_Curr_Empr': Time_With_Curr_Empr,
        'Credit_Score': Credit_Score
    }
    
    features = pd.DataFrame(data, index=[0])
    features = pd.get_dummies(features)

    trained_columns = pickle.load(open("trained_columns.pkl", "rb"))
    features = features.reindex(columns=trained_columns, fill_value=0)

    return features

input_df = user_input_features()

if st.button("Predict"):

    # Map numeric to labels
    class_mapping = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}

    prediction = model.predict(input_df)[0]
    prediction_label = class_mapping[prediction]

    prediction_proba = model.predict_proba(input_df)[0]
    prob_df = pd.DataFrame({
        'Category': [class_mapping[c] for c in model.classes_],
        'Probability': prediction_proba
    })

    # Show only label + bar chart
    st.subheader("Predicted Approved Flag")
    st.write(f"**{prediction_label}**")

    st.subheader("Prediction Probabilities")
    st.bar_chart(prob_df.set_index('Category'))