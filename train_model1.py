import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.utils import resample
import pickle

data = pd.read_csv("gps_data1.csv")

data['speed'] = pd.to_numeric(data['speed'], errors='coerce')
data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')

data['latitude'] = data['latitude'].clip(-90, 90)

data['lat_shift'] = data['latitude'].shift(1)
data['lon_shift'] = data['longitude'].shift(1)
data['speed_shift'] = data['speed'].shift(1)

data['lat_shift'] = data['lat_shift'].fillna(0)
data['lon_shift'] = data['lon_shift'].fillna(0)
data['speed_shift'] = data['speed_shift'].fillna(0)

def safe_geodesic(row):
    try:
        return geodesic((row['latitude'], row['longitude']),
                        (row['lat_shift'], row['lon_shift'])).meters
    except ValueError:
        return 0

data['distance'] = data.apply(safe_geodesic, axis=1)

data['speed_change'] = data['speed'] - data['speed_shift']
data['acceleration'] = data['speed_change'] / (data['distance'] + 1e-3)

data['label'] = 0  

data.loc[(data['distance'] > 1000) | (data['speed'] < 2), 'label'] = 1

data.loc[(data['speed'] > 120) | (data['acceleration'] > 15), 'label'] = 2


data = data.dropna()

normal_data = data[data['label'] == 0]
spoofing_data = data[data['label'] == 1]
jamming_data = data[data['label'] == 2]

spoofing_data_downsampled = spoofing_data.sample(n=int(len(normal_data) * 0.2), random_state=42)
jamming_data_downsampled = jamming_data.sample(n=int(len(normal_data) * 0.2), random_state=42)

data_balanced = pd.concat([normal_data, spoofing_data_downsampled, jamming_data_downsampled])

data_balanced['latitude'] += np.random.normal(0, 0.01, size=len(data_balanced))
data_balanced['longitude'] += np.random.normal(0, 0.01, size=len(data_balanced))
data_balanced['speed'] += np.random.normal(0, 1.0, size=len(data_balanced))

flip_indices = np.random.choice(data_balanced.index, size=int(len(data_balanced) * 0.05), replace=False)
data_balanced.loc[flip_indices, 'label'] = np.random.choice([0, 1, 2], size=len(flip_indices))

X = data_balanced[['latitude', 'longitude', 'speed']]
y = data_balanced['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=42,
    eval_metric="mlogloss"
)

hybrid_model = VotingClassifier(
    estimators=[('random_forest', rf_model), ('xgboost', xgb_model)],
    voting='soft'
)

hybrid_model.fit(X_train, y_train)

y_pred = hybrid_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Hybrid Model Accuracy: {accuracy * 100:.2f}%")

with open("hybrid_gps_detection_model1.pkl", "wb") as f:
    pickle.dump(hybrid_model, f)

if __name__ == "__main__":
    with open("hybrid_gps_detection_model1.pkl", "rb") as f:
        hybrid_model = pickle.load(f)
