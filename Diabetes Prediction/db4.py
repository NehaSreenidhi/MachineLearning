import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Define the BMI and Insulin category functions
def bmi_category(bmi):
    if bmi <= 18.5:
        return 'Underweight'
    elif 18.5 < bmi <= 24.9:
        return 'Healthy'
    elif 25 <= bmi <= 29.9:
        return 'Pre-obese'
    elif 30 <= bmi <= 34.9:
        return 'Obesity class 1'
    elif 35 <= bmi <= 39.9:
        return 'Obesity class 2'
    else:
        return 'Obesity class 3'

def insulin_category(insulin):
    if insulin < 25:
        return 'Fasting or 3+ hours after glucose ingestion'
    elif 25 <= insulin < 30:
        return '30 minutes after glucose administration'
    elif 30 <= insulin <= 230:
        return '1 hour after glucose ingestion'
    elif 230 < insulin <= 276:
        return '2 hours after glucose ingestion'
    else:
        return 'Other'

# Streamlit app title
st.title("Diabetes Prediction by ")
st.subheader("Predict the likelihood of diabetes based on health metrics")
st.markdown("**Take control of your health today!** Use this tool to determine your diabetes risk based on various health indicators.")

# Input form
with st.form("input_form"):
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0, help="Number of times pregnant")
    glucose = st.number_input('Glucose', min_value=0, max_value=300, value=0, help="Plasma glucose concentration")
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200, value=0, help="Diastolic blood pressure (mm Hg)")
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=0, help="Triceps skin fold thickness (mm)")
    insulin = st.number_input('Insulin', min_value=0, max_value=1000, value=0, help="2-Hour serum insulin (mu U/ml)")
    bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=0.0, help="Body mass index (weight in kg/(height in m)^2)")
    diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.0, help="Diabetes pedigree function")
    age = st.number_input('Age', min_value=0, max_value=120, value=0, help="Age (years)")

    # Submit button
    submit_button = st.form_submit_button(label='Predict')

# Create DataFrame from input
if submit_button:
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree_function],
        'Age': [age]
    })

    # Process the input data
    input_data['BMI_Category'] = input_data['BMI'].apply(bmi_category)
    input_data['Insulin_Category'] = input_data['Insulin'].apply(insulin_category)
    input_data = pd.get_dummies(input_data, columns=['BMI_Category', 'Insulin_Category'])

    # Define feature names used during training
    feature_names = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age',
        'BMI_Category_Healthy', 'BMI_Category_Obesity class 1', 'BMI_Category_Obesity class 2',
        'BMI_Category_Obesity class 3', 'BMI_Category_Pre-obese', 'BMI_Category_Underweight',
        'Insulin_Category_1 hour after glucose ingestion',
        'Insulin_Category_30 minutes after glucose administration',
        'Insulin_Category_Fasting or 3+ hours after glucose ingestion',
        'Insulin_Category_Other'
    ]

    # Ensure the input data has the same feature columns as the model
    for feature in feature_names:
        if feature not in input_data.columns:
            input_data[feature] = 0
    input_data = input_data[feature_names]

    # Load the model
    model_filename = 'svm_model_diabetes.pkl'
    try:
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        st.error(f"Model file {model_filename} not found.")
        model = None

    # Predict and display the result
    if model:
        # Standardize numerical columns
        numerical_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        scaler = StandardScaler()
        input_data[numerical_columns] = scaler.fit_transform(input_data[numerical_columns])

        # Make predictions
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("Prediction: Diabetic")
            st.balloons()
        else:
            st.success("Prediction: Not Diabetic")
            st.snow()
    else:
        st.error("Model is not loaded. Please check the model file.")


