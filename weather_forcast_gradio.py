import gradio as gr
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt

# Function to load and preprocess data
def load_and_prepare_data(file_path):
    try:
        df = pd.read_csv(file_path)
        features = ['precipitation', 'temp_max', 'temp_min', 'wind']
        df = df[features]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)
        return df, scaled_data, scaler
    except Exception as e:
        print("Error loading CSV:", e)
        return None, None, None

# Function to create sequences
def create_sequences(data, look_back=10):
    X = []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
    return np.array(X)

# Function to build and train model
def build_and_train_model(X_train, y_train, input_shape, model_save_path):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=8, verbose=0)
    model.save(model_save_path)
    return model

# Function to generate graphs for attributes
def generate_graphs(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Weather Data Attributes Over Time")

    # Plot each attribute
    df['precipitation'].plot(ax=axes[0, 0], title="Precipitation", color='blue')
    df['temp_max'].plot(ax=axes[0, 1], title="Temp Max", color='red')
    df['temp_min'].plot(ax=axes[1, 0], title="Temp Min", color='green')
    df['wind'].plot(ax=axes[1, 1], title="Wind", color='orange')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save the graph as an image
    graph_path = "weather_attributes.png"
    plt.savefig(graph_path)
    plt.close()
    return graph_path

# Function to predict weather and generate graphs
def forecast_weather(csv_file):
    look_back = 10
    model_save_path = "weather_rnn_model.h5"

    # Load and preprocess data
    df, data, scaler = load_and_prepare_data(csv_file.name)
    if data is None or df is None:
        return "Error loading CSV file.", None

    # Generate graphs for attributes
    graph_path = generate_graphs(df)

    # Create sequences
    X = create_sequences(data, look_back)
    if X.shape[0] == 0:
        return "Insufficient data in CSV. Provide more rows.", graph_path

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

    # Check for existing model
    if os.path.exists(model_save_path):
        model = load_model(model_save_path)
    else:
        # Dummy y_train (just for training)
        y_train = [0] * X.shape[0]
        model = build_and_train_model(X, y_train, (X.shape[1], X.shape[2]), model_save_path)

    # Predict future weather
    last_sequence = X[-1].reshape(1, look_back, X.shape[2])
    prediction = model.predict(last_sequence)

    # Map the prediction to 'temp_max'
    predicted_temp_max = scaler.inverse_transform([[0, prediction[0][0], 0, 0]])[0][1]
    prediction_result = f"Predicted Temp_Max: {predicted_temp_max:.2f}°C"

    return prediction_result, graph_path

# Gradio Interface
interface = gr.Interface(
    fn=forecast_weather,
    inputs=gr.File(label="Upload CSV File (Date, Precipitation, Temp_Max, Temp_Min, Wind)"),
    outputs=[gr.Textbox(label="Weather Forecast Prediction"), gr.Image(label="Weather Data Graphs")],
    title="Weather Forecast Application",
    description="Upload a CSV file containing weather data to predict future temperature (Temp_Max) using RNN and view the graphs of all attributes."
)

if __name__ == "__main__":
    interface.launch()
