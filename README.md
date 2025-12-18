Problem: Voyage AI – Travel Delay Prediction

This project trains a Gradient Boosting (XGBoost) model
to predict travel delays using synthetic travel data.

Input:
- Distance
- Transport type
- Time
- Day
- Ticket price
- Past delay

Output:
- Delay in minutes

The final trained model is exported in ONNX format
as required by the hackathon.

Files:
- train.py → training code
- inference.py → ONNX inference
- travel_delay_model.onnx → final model
