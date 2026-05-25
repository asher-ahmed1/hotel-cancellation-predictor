# 🏨 Hotel Booking Cancellation Predictor
### DL Lab Final Project

Predicts whether a hotel booking will be cancelled using a Deep Neural Network.

## Model Results
- **Test Accuracy:** 78.00%
- **ROC-AUC:** 0.8408

## Architecture
Deep MLP: 256 → 128 → 64 → 32 → 1 (Sigmoid)

## DL Concepts Applied
- Batch Normalization
- Dropout Regularization
- Binary Cross-Entropy Loss
- Adam Optimizer
- Early Stopping
- ReduceLROnPlateau

## Tech Stack
Flask · TensorFlow/Keras · scikit-learn · pandas · HTML/CSS/JS

## How to Run
1. Clone the repo
2. Add `hotel_bookings.csv` to the project folder
3. Install dependencies:
   pip install flask tensorflow scikit-learn pandas numpy
4. Run:
   python app.py
5. Open http://localhost:5000
