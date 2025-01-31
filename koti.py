import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Define datasets for different cities
city_datasets = {
    "Hyderabad": "hyd.csv",
    "Bangalore": "bangalore.csv",
    "Mumbai": "mumbai.csv",
    "Delhi": "delhi.csv",
    "Chennai": "chennai.csv",
    "Kolkata": "kolkata.csv",
}

# Streamlit UI
st.title("House Price Prediction")

# Select city
selected_city = st.selectbox("Select a City", list(city_datasets.keys()))

# Function to train the model for a selected city
@st.cache_resource
def train_city_model(city):
    """Train and return model, encoders, and scalers for the selected city."""
    df = pd.read_csv(city_datasets[city])

    # Identify categorical and numerical columns
    target_column = "Price"
    categorical_columns = ['Location', 'Resale', 'MaintenanceStaff', 'Gymnasium', 'LandscapedGardens',
                           'JoggingTrack', 'RainWaterHarvesting', 'IndoorGames', 'ShoppingMall', 'Intercom',
                           'SportsFacility', 'ATM', 'ClubHouse', 'School', '24X7Security', 'PowerBackup',
                           'CarParking', 'StaffQuarter', 'Cafeteria', 'MultipurposeRoom', 'Hospital', 'Wifi',
                           "Children'splayarea", 'LiftAvailable', 'VaastuCompliant', 'GolfCourse']
    numerical_columns = ['Area', 'No. of Bedrooms']

    # Clean up categorical columns
    for col in categorical_columns:
        df[col] = df[col].replace(9, np.nan)  # Replace invalid entries with NaN
        df[col] = df[col].fillna(df[col].mode()[0])  # Fill missing values with the mode

    # Convert binary columns to "Yes"/"No"
    binary_columns = ['Gymnasium', 'Resale', 'MaintenanceStaff', 'RainWaterHarvesting', 'IndoorGames',
                      'ShoppingMall', 'Intercom', 'SportsFacility', 'ATM', 'ClubHouse', 'School', '24X7Security',
                      'PowerBackup', 'CarParking', 'StaffQuarter', 'Cafeteria', 'MultipurposeRoom', 'Hospital',
                      'Wifi', "Children'splayarea", 'LiftAvailable', 'VaastuCompliant', 'GolfCourse',
                      'LandscapedGardens', 'JoggingTrack']
    for col in binary_columns:
        df[col] = df[col].map({1: "Yes", 0: "No"})

    # Label encode categorical columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Split data into features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize numerical columns
    scaler = StandardScaler()
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

    # Train the RandomForest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model performance
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    return model, scaler, label_encoders, mae, numerical_columns, categorical_columns

# Get the trained model and resources for the selected city
model, scaler, label_encoders, mae, numerical_columns, categorical_columns = train_city_model(selected_city)

# Display the model's performance
st.write(f"*Model Accuracy (Mean Absolute Error) for {selected_city}: \u20b9{mae:,.2f}*")

st.header("Enter House Details")

# User inputs for numerical features
user_inputs = {}
for col in numerical_columns:
    user_inputs[col] = st.number_input(f"{col} (numeric input)", min_value=1, step=1)

# User inputs for categorical features
for col in categorical_columns:
    if col in label_encoders:  # Handle binary columns
        if col in ['Gymnasium', 'Resale', 'MaintenanceStaff', 'RainWaterHarvesting', 'IndoorGames', 'ShoppingMall',
                   'Intercom', 'SportsFacility', 'ATM', 'ClubHouse', 'School', '24X7Security', 'PowerBackup',
                   'CarParking', 'StaffQuarter', 'Cafeteria', 'MultipurposeRoom', 'Hospital', 'Wifi',
                   "Children'splayarea", 'LiftAvailable', 'VaastuCompliant', 'GolfCourse', 'LandscapedGardens',
                   'JoggingTrack']:
            col_value = st.radio(f"{col}", ['Yes', 'No'], horizontal=True)
            user_inputs[col] = col_value
        else:
            user_inputs[col] = st.selectbox(f"Select {col}", label_encoders[col].classes_)

# Predict button
if st.button("Predict Price"):
    # Encode categorical inputs
    for col in categorical_columns:
        if col in ['Gymnasium', 'Resale', 'MaintenanceStaff', 'RainWaterHarvesting', 'IndoorGames', 'ShoppingMall',
                   'Intercom', 'SportsFacility', 'ATM', 'ClubHouse', 'School', '24X7Security', 'PowerBackup',
                   'CarParking', 'StaffQuarter', 'Cafeteria', 'MultipurposeRoom', 'Hospital', 'Wifi',
                   "Children'splayarea", 'LiftAvailable', 'VaastuCompliant', 'GolfCourse', 'LandscapedGardens',
                   'JoggingTrack']:
            user_inputs[col] = 1 if user_inputs[col] == "Yes" else 0
        else:
            user_inputs[col] = label_encoders[col].transform([user_inputs[col]])[0]

    # Normalize numerical inputs
    scaled_values = scaler.transform([[user_inputs[col] for col in numerical_columns]])[0]

    # Prepare input data
    input_data = {**user_inputs, numerical_columns[0]: scaled_values[0], numerical_columns[1]: scaled_values[1]}
    input_df = pd.DataFrame([input_data])

    # Ensure columns are in the correct order
    input_df = input_df[model.feature_names_in_]

    # Predict price
    predicted_price = model.predict(input_df)[0]

    # Display prediction
    st.success(f" Predicted House Price in {selected_city}: {predicted_price:,.2f}")
