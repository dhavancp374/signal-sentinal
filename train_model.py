import pandas as pd
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import random

# Load dataset
data = pd.read_csv("gps_data.csv")

# Function to validate latitude and longitude
def validate_coordinates(row):
    # Latitude must be between -90 and 90, longitude between -180 and 180
    if row['latitude'] < -90:
        row['latitude'] = -90
    elif row['latitude'] > 90:
        row['latitude'] = 90

    if row['longitude'] < -180:
        row['longitude'] = -180
    elif row['longitude'] > 180:
        row['longitude'] = 180

    return row

# Apply validation to each row
data = data.apply(validate_coordinates, axis=1)

# Feature engineering
data['lat_shift'] = data['latitude'].shift(1)
data['lon_shift'] = data['longitude'].shift(1)
data['speed_shift'] = data['speed'].shift(1)

# Calculate distance, speed change, and acceleration
data['distance'] = data.apply(
    lambda row: geodesic((row['latitude'], row['longitude']), 
                         (row['lat_shift'], row['lon_shift'])).meters 
    if not pd.isnull(row['lat_shift']) else 0, axis=1
)
data['speed_change'] = data['speed'] - data['speed_shift']
data['acceleration'] = data['speed_change'] / (data['distance'] + 1e-3)

# Labeling based on conditions
data['label'] = 0  # Default label is 0 (Normal)
data.loc[(data['distance'] > 1000) | (data['speed'] < 2), 'label'] = 1  # Label as spoofing
data.loc[(data['speed'] > 120) | (data['acceleration'] > 15), 'label'] = 2  # Label as jamming

# Drop rows with NaN values
data = data.dropna()

# Features and labels
X = data[['latitude', 'longitude', 'speed', 'distance', 'speed_change', 'acceleration']]
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# RandomForest model with updated hyperparameters
model = RandomForestClassifier(
    random_state=42,
    max_depth=2,  # Maximum depth of trees
    n_estimators=20,  # Number of trees in the forest
    min_samples_split=20,  # Minimum samples required to split a node
    min_samples_leaf=10,  # Minimum samples required to be at a leaf node
    max_samples=0.8,  # Max samples used for building each tree
    bootstrap=True  # Whether bootstrap samples are used when building trees
)

# Train the model
model.fit(X_train, y_train)
import pandas as pd
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import random

# Load dataset
data = pd.read_csv("gps_datareal4.csv")

# Function to validate latitude and longitude
def validate_coordinates(row):
    # Latitude must be between -90 and 90, longitude between -180 and 180
    if row['latitude'] < -90:
        row['latitude'] = -90
    elif row['latitude'] > 90:
        row['latitude'] = 90

    if row['longitude'] < -180:
        row['longitude'] = -180
    elif row['longitude'] > 180:
        row['longitude'] = 180

    return row

# Apply validation to each row
data = data.apply(validate_coordinates, axis=1)

# Feature engineering
data['lat_shift'] = data['latitude'].shift(1)
data['lon_shift'] = data['longitude'].shift(1)
data['speed_shift'] = data['speed'].shift(1)

# Calculate distance, speed change, and acceleration
data['distance'] = data.apply(
    lambda row: geodesic((row['latitude'], row['longitude']), 
                         (row['lat_shift'], row['lon_shift'])).meters 
    if not pd.isnull(row['lat_shift']) else 0, axis=1
)
data['speed_change'] = data['speed'] - data['speed_shift']
data['acceleration'] = data['speed_change'] / (data['distance'] + 1e-3)

# Labeling based on conditions
data['label'] = 0  # Default label is 0 (Normal)
data.loc[(data['distance'] > 1000) | (data['speed'] < 2), 'label'] = 1  # Label as spoofing
data.loc[(data['speed'] > 120) | (data['acceleration'] > 15), 'label'] = 2  # Label as jamming

# Drop rows with NaN values
data = data.dropna()

# Features and labels
X = data[['latitude', 'longitude', 'speed', 'distance', 'speed_change', 'acceleration']]
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42, stratify=y)

# RandomForest model with updated hyperparameters
model = RandomForestClassifier(
    random_state=42,
    max_depth=5,  # Maximum depth of trees
    n_estimators=50,  # Number of trees in the forest
    min_samples_split=10,  # Minimum samples required to split a node
    min_samples_leaf=5,  # Minimum samples required to be at a leaf node
    max_samples=0.8,  # Max samples used for building each tree
    bootstrap=True,  # Whether bootstrap samples are used when building trees
    class_weight='balanced'  # Class weights to handle imbalance
)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
with open("gps_detection_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Predict and alert function
def predict_and_alert(model, input_features):
    prediction = model.predict(input_features)
    for pred in prediction:
        if pred == 0:
            print("Alert: Normal activity detected.")
        elif pred == 1:
            print("Alert: Potential GPS spoofing detected!")
        elif pred == 2:
            print("Alert: Potential GPS jamming detected!")
    return prediction

# Example for testing the model
if __name__ == "__main__":
    with open("gps_detection_model.pkl", "rb") as f:
        model = pickle.load(f)

    test_input = pd.DataFrame([{
        'latitude': 13.7275,
        'longitude': 80.2298,
        'speed': 150.0,
        'distance': 600.0,
        'speed_change': 50.0,
        'acceleration': 12.0
    }])

    predict_and_alert(model, test_input)
# Predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
with open("gps_detection_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Predict and alert function
def predict_and_alert(model, input_features):
    prediction = model.predict(input_features)
    for pred in prediction:
        if pred == 0:
            print("Alert: Normal activity detected.")
        elif pred == 1:
            print("Alert: Potential GPS spoofing detected!")
        elif pred == 2:
            print("Alert: Potential GPS jamming detected!")
    return prediction

# Example for testing the model
if __name__ == "__main__":
    with open("gps_detection_model.pkl", "rb") as f:
        model = pickle.load(f)

    test_input = pd.DataFrame([{
        'latitude': 13.7275,
        'longitude': 80.2298,
        'speed': 150.0,
        'distance': 600.0,
        'speed_change': 50.0,
        'acceleration': 12.0
    }])

    predict_and_alert(model, test_input)
