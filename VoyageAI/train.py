import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

df = pd.read_csv("vande_bharat.csv")

print("Columns in dataset:")
print(df.columns)

df = df[['Distance', 'Travel Time']]
df.columns = ['distance_raw', 'travel_time_raw']

def parse_distance(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower()
    match = re.search(r'([\d.]+)\s*km', s)
    return float(match.group(1)) if match else np.nan


df['distance_km'] = df['distance_raw'].apply(parse_distance)


def parse_travel_time(val):
    if pd.isna(val):
        return np.nan

    s = str(val).lower()
    s = re.sub(r'\(.*?\)', '', s)  # remove (Monsoon)

    h = re.search(r'(\d+)\s*h', s)
    m = re.search(r'(\d+)\s*m', s)

    hours = int(h.group(1)) if h else 0
    minutes = int(m.group(1)) if m else 0

    total = hours * 60 + minutes
    return total if total > 0 else np.nan


df['travel_time'] = df['travel_time_raw'].apply(parse_travel_time)


df = df[['distance_km', 'travel_time']]
df = df.dropna()

print("Rows after cleaning:", len(df))
print(df.head())
print(df.dtypes)

if len(df) == 0:
    raise ValueError("Dataset empty after parsing.")


X = df[['distance_km']]
y = df['travel_time']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training complete")



predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error (minutes):", mae)



initial_type = [("input", FloatTensorType([None, 1]))]

onnx_model = convert_sklearn(
    model,
    initial_types=initial_type
)

with open("vande_bharat_travel_time.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("ONNX model saved successfully")
