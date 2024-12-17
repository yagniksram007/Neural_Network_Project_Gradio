import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model  # Import load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

# Step 1: Read the CSV file
def load_and_prepare_data(file_path):
    try:
        # Load data
        df = pd.read_csv(file_path)
        print("Data Loaded Successfully!")
        
        # Drop date column for simplicity and focus on numeric values
        features = ['precipitation', 'temp_max', 'temp_min', 'wind']
        df = df[features]
        
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)
        
        return scaled_data, scaler
    except Exception as e:
        print("Error loading CSV file:", e)
        return None, None

# Step 2: Create data sequences
def create_sequences(data, look_back=10):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i + look_back][1])  # Predicting 'temp_max' as an example
    return np.array(X), np.array(y)

# Step 3: Build and train the RNN model
def build_and_train_model(X_train, y_train, input_shape, model_save_path):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))  # Output one value (e.g., 'temp_max')
    model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=8, verbose=1)
    
    # Save the model to the specified path
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    return model

# Step 4: Load an existing model
def load_existing_model(model_path):
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = load_model(model_path)
        return model
    else:
        print(f"Model file {model_path} does not exist.")
        return None

# Step 5: Main function to run the process
def predict_weather(file_path):
    look_back = 10
    model_save_path = "weather_rnn_model.h5"

    # Load and preprocess data
    data, scaler = load_and_prepare_data(file_path)
    if data is None:
        return

    # Create sequences
    X, y = create_sequences(data, look_back)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

    # Check if model exists
    if os.path.exists(model_save_path):
        model = load_existing_model(model_save_path)
    else:
        print("Training a new RNN model...")
        model = build_and_train_model(X, y, (X.shape[1], X.shape[2]), model_save_path)
    
    # Predict future weather
    last_sequence = X[-1].reshape(1, look_back, X.shape[2])
    prediction = model.predict(last_sequence)
    
    # Inverse transform the prediction
    predicted_value = scaler.inverse_transform([[0, prediction[0][0], 0, 0]])[0][1]  # Map to 'temp_max'
    print(f"Predicted Temp_Max: {predicted_value:.2f}°C")

# Run the function with a sample CSV file
if __name__ == "__main__":
    file_path = "CSV_Files/seattle-weather.csv"  
    predict_weather(file_path)
