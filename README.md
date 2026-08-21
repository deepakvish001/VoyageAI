## VoyageAI – Vande Bharat Travel Time Prediction

Trains a `scikit-learn` `GradientBoostingRegressor` to predict a Vande
Bharat Express train's total travel time from its route distance, using
real route data (`VoyageAI/vande_bharat.csv`) rather than synthetic data.
The trained model is exported to ONNX so inference doesn't need
scikit-learn installed.

**Input:** distance, in kilometres.

**Output:** predicted total travel time, in minutes.

Mean absolute error on the held-out test split is about 51 minutes — the
dataset is 42 routes after cleaning, so treat this as a baseline rather
than a tuned model.

### Files

- `VoyageAI/train.py` — parses `vande_bharat.csv`, trains the model, and
  exports it to `VoyageAI/vande_bharat_travel_time.onnx`.
- `VoyageAI/inference.py` — loads that ONNX model and runs one sample
  prediction.
- `VoyageAI/requirements.txt` — pinned dependencies for both scripts.

### Run it

```
pip install -r VoyageAI/requirements.txt
cd VoyageAI
python train.py       # optional — vande_bharat_travel_time.onnx is already committed
python inference.py
```
