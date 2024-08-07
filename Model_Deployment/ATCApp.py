import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder


# load the model
# model = pickle.load(open('new_xgb_model.pkl', 'rb'))
import os
model_path = os.path.join(os.getcwd(), 'new_xgb_model.pkl')
model = pickle.load(open(model_path, 'rb'))

# categorical features
categorical_features = {
    'REGION':['DAKAR','SAINT-LOUIS','TAMBACOUNDA','FATICK','THIES', 'LOUGA', 'KAFFRINE', 'DIOURBEL', 'KOLDA','MATAM', 'SEDHIOU', 'KAOLACK', 'ZIGUINCHOR', 'KEDOUGOU']
}

# define ecndoder dictionary
encoder_dict = {feature: LabelEncoder().fit(values) for feature, values in categorical_features.items()}

# ==================================================================
# Streamlit app configuration
st.set_page_config(page_title="Customer Churn Prediction", page_icon=":bar_chart:", layout="centered", initial_sidebar_state="expanded")

# Adding a banner image
# st.image("connected.jpg", use_column_width=True)

# title and header
st.title('Telco. Customer Churn Prediction')
st.sidebar.header('Customer Details')

input_data = {}

# creating input boxes for each feature: [ ['REGION', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 'REGULARITY', 'TENURE_LE']]

# '''we use only 8 features since they are the top 8 features
# Important Note: This app does capture comprehensive data of each client.
# Future work must include fields to capture other data inputs for purposes of future projects
# '''
# ---------------------------------------------------------


# input_data['user_id'] = st.sidebar.text_input('customer ID')
input_data['REGION'] = st.sidebar.selectbox('REGION', options=categorical_features['REGION'])
input_data['MONTANT'] = st.sidebar.number_input('MONTANT: Top-Up Amount')
input_data['FREQUENCE_RECH'] = st.sidebar.number_input('Number of Times Client Refilled')
input_data['REVENUE'] = st.sidebar.number_input('Revenue')
input_data['ARPU_SEGMENT'] = st.sidebar.number_input('Income over 90 Days')
input_data['FREQUENCE'] = st.sidebar.number_input('Number of Times Client Made an income')
input_data['REGULARITY'] = st.sidebar.number_input('Regularity')
input_data['TENURE_LE'] = st.sidebar.number_input('Average Duration in Network (Months)', value=9.0, step=0.5)

# convert input data into dataframe
input_df = pd.DataFrame([input_data])



# Encode categorical feature
for feature, encoder in encoder_dict.items():
    input_df[feature] = encoder.transform(input_df[feature])




# Creating the Predict button
if st.button('PREDICT'):
    try:
        prediction = model.predict(input_df)
        if prediction[0] == 1:
            # st.write('The customer is likely to churn.')
            st.markdown("<h3 style='color: red;'>The customer is likely to churn.</h3>", unsafe_allow_html=True)
        else:
            # st.write('The customer is not likely to churn.')
            st.markdown("<h3 style='color: green;'>The customer is not likely to churn.</h3>", unsafe_allow_html=True)
    except Exception as e:
        st.write("Error during prediction:", str(e))


# Additional sections
st.markdown("## How It Works")
st.write("""
The Telco Customer Churn Prediction app allows you to input customer details and predict the likelihood of churn.
Simply enter the required details in the sidebar and click the 'PREDICT' button.
""")