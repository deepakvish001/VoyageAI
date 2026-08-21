import argparse

import onnxruntime as rt
import numpy as np

parser = argparse.ArgumentParser(description="Predict Vande Bharat travel time from distance.")
parser.add_argument(
    "distance_km",
    nargs="?",
    type=float,
    default=500,
    help="Route distance in kilometers (default: 500)",
)
args = parser.parse_args()

session = rt.InferenceSession("vande_bharat_travel_time.onnx")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Model loaded successfully")
print("Input name:", input_name)
print("Output name:", output_name)

sample_input = np.array([[args.distance_km]], dtype=np.float32)

prediction = session.run(
    [output_name],
    {input_name: sample_input}
)

predicted_minutes = prediction[0].item()
predicted_hours = predicted_minutes / 60

print(f"\nPredicted Travel Time for {args.distance_km:g} km:")
print(f"{predicted_minutes:.2f} minutes")
print(f"{predicted_hours:.2f} hours")
