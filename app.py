from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from geopy.distance import geodesic
import pickle
import numpy as np
import random
import csv

app = Flask(__name__)
CORS(app)

# Load the trained model
with open("gps_detection_model.pkl", "rb") as f:
    model = pickle.load(f)

# Read GPS data from a CSV file
csv_file_path = "gps_datareal4.csv"
gps_data_list = []
with open(csv_file_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        gps_data_list.append({
            "lat": float(row["latitude"]),
            "lon": float(row["longitude"]),
            "speed": float(row["speed"])
        })

# Index to keep track of current data being streamed
current_index = 0
data_log = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/gps-data', methods=['GET'])
def get_gps_data():
    global current_index, gps_data_list

    if current_index >= len(gps_data_list):
        return jsonify({"error": "No more data to stream."}), 404

    gps_data = gps_data_list[current_index]
    current_index += 1  # Increment to the next data point

    return jsonify(gps_data)

def detect_anomaly_and_jamming(new_point):
    random_offset_lat = random.uniform(-0.0001, 0.0001)
    random_offset_lon = random.uniform(-0.0001, 0.0001)
    random_speed_offset = random.uniform(-1, 1)

    adjusted_lat = new_point["lat"] + random_offset_lat
    adjusted_lon = new_point["lon"] + random_offset_lon
    adjusted_speed = new_point["speed"] + random_speed_offset

    # Calculate distance and speed change
    distance = geodesic((adjusted_lat, adjusted_lon), (new_point["lat"], new_point["lon"])).meters
    speed_change = new_point["speed"] - adjusted_speed
    acceleration = speed_change / (distance + 1e-3)  # Avoid division by zero

    print(f"Distance: {distance}, Speed Change: {speed_change}, Acceleration: {acceleration}")

    # Artificially introduce anomalies to simulate spoofing and jamming
    if random.uniform(0, 1) < 0.2:  
        distance *= random.uniform(5, 10)  
        acceleration *= random.uniform(2, 5)  

    # Predict anomaly using the model
    features = np.array([[new_point["lat"], new_point["lon"], new_point["speed"],
                          distance, speed_change, acceleration]])

    label = model.predict(features)[0]
    print(f"Model predicted label: {label}")

    # Default alert message
    alert_message = "Normal activity detected."
    if label == 1:  # Spoofing detected
        alert_message = "Alert: Potential GPS spoofing detected!"
    elif label == 2:  # Jamming detected
        alert_message = "Alert: Potential GPS jamming detected!"

    print(f"Alert message: {alert_message}")

    return {
        "is_anomaly": bool(label == 1 or label == 2),
        "is_jamming": bool(label == 2),
        "alert_message": alert_message,
        "features": features.tolist()
    }

def make_serializable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()  
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  
    return obj  

@app.route('/detect-anomaly', methods=['POST'])
def detect_anomaly():
    global data_log

    new_point = request.json
    print(f"Received new point: {new_point}")

    detection_result = detect_anomaly_and_jamming(new_point)
    print(f"Detection result: {detection_result}")

    data_log.append({"new_point": new_point, "detection_result": detection_result})
    serializable_result = {k: make_serializable(v) for k, v in detection_result.items()}

    print(f"Serialized detection result sent to frontend: {serializable_result}")

    return jsonify(serializable_result)

if __name__ == "__main__":
    app.run(debug=True)
