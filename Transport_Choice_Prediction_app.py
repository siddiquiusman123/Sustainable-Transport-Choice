import pandas as pd
import numpy as np
import joblib 
import streamlit as st


st.title("🚇 Transport Choice Prediction App")

# Load The model , encoder , scalar

model = joblib.load("transport_choice_prd.pkl")
ordinal_encoder = joblib.load("transport_choice_ord_encoder.pkl")
label_encoder = joblib.load("transport_choice_label_encoder.pkl")
scalar = joblib.load("transport_choice_scalar.pkl")

# User input 

age = st.number_input("Enter Your Age")
gender = st.radio("Select Gender",["Male","Female"])
trip_distance = st.number_input("Enter Trip Distance (in km)")
travel_time = st.number_input("Enter Travel Time (in Minutes)")
travel_cost = st.number_input("Enter Travel Cost (Monthly INR)")
vehicle_in_household = st.radio("Vehicles in Household",[0,1,2,3,4])
public_transport_avaiable = st.radio("Public Transport Available",['Yes', 'No'])
cost_sensitivity = st.radio("Cost Sensitivity",['Low','Medium','High'])
comfort_preference = st.radio("Comfort Preference",['Low','Medium','High'])
environmental_concern = st.radio("Environmental Concern",['Low','Medium','High'])
occupatoin = st.radio("Occupation",['Student', 'Professional', 'Worker', 'Retired'])



# Age	Gender	Trip Distance (in km)	Travel Time (in Minutes)	Travel Cost (Monthly INR)	
# Vehicles in Household	Public Transport Available
# Cost Sensitivity	Comfort Preference	Environmental Concern	Occupation	Mode

input_data = [[age , gender , trip_distance , travel_time , travel_cost ,
            vehicle_in_household , public_transport_avaiable ,
            cost_sensitivity , comfort_preference , environmental_concern , occupatoin]]

# Data Frame of Input Data

input_df = pd.DataFrame(input_data , columns=['Age', 'Gender', 'Trip Distance (in km)', 
                                              'Travel Time (in Minutes)',
                                               'Travel Cost (Monthly INR)',
                                               'Vehicles in Household', 
                                               'Public Transport Available',
                                               'Cost Sensitivity',
                                               'Comfort Preference', 
                                               'Environmental Concern',
                                               'Occupation'])
# print(input_df)

# Encoding Chategorical Column
chategorical_columns = input_df.select_dtypes(include=['object','bool']).columns
encoded = ordinal_encoder.transform(input_df[chategorical_columns])
enc_df = pd.DataFrame(encoded, columns=chategorical_columns, index=input_df.index)
final_input = pd.concat([input_df.drop(columns=chategorical_columns, axis=1), enc_df], axis=1)

# Scaling Numeric Columns

numeric_columns = input_df.select_dtypes(include=['int64','float64']).columns
final_input[numeric_columns] = scalar.transform(final_input[numeric_columns])

# Predict Button

if st.button("Predict"):
    prediction = model.predict(final_input)
    prediction_orignal = label_encoder.inverse_transform(prediction)
    st.success(prediction_orignal[0])


# # --- Encode categorical columns ---
# categorical_columns = input_df.select_dtypes(include=['object','bool']).columns
# encoded = ordinal_encoder.transform(input_df[categorical_columns])
# enc_df = pd.DataFrame(encoded, columns=categorical_columns, index=input_df.index)

# # Merge numeric + categorical
# final_input = pd.concat([input_df.drop(columns=categorical_columns), enc_df], axis=1)

# # --- Scale numeric columns (use same columns as training) ---
# numeric_columns = scalar.feature_names_in_   # scaler remembers training columns
# final_input[numeric_columns] = scalar.transform(final_input[numeric_columns])

# # --- Reorder columns to match training ---
# final_input = final_input[model.feature_names_in_]
