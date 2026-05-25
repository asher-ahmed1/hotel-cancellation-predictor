"""
Flask Web App — Hotel Booking Cancellation Predictor (Deep Learning)
Run: python app.py
Visit: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import os

app = Flask(__name__)

# ── Global model state ──────────────────────────────────────────
model = None
scaler = None
accuracy = 0.0
auc = 0.0
features = [
    'lead_time', 'total_guests', 'total_nights',
    'previous_cancellations', 'previous_bookings_not_canceled',
    'booking_changes', 'required_car_parking_spaces',
    'total_of_special_requests', 'adr',
    'revenue_estimate', 'guest_per_night',
    'has_previous_cancel', 'is_repeated_guest',
    'days_in_waiting_list',
]

def build_model(input_dim):
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(256)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inputs=inp, outputs=out)
    m.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return m

def train():
    global model, scaler, accuracy, auc
    print("📂 Loading dataset …")
    df = pd.read_csv("hotel_bookings.csv")
    df['children'] = df['children'].fillna(0)
    df['agent']    = df['agent'].fillna(0)
    df['country']  = df['country'].fillna('Unknown')
    df['total_guests']        = df['adults'] + df['children'] + df['babies']
    df['total_nights']        = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['revenue_estimate']    = df['adr'] * df['total_nights'].replace(0, 1)
    df['guest_per_night']     = df['total_guests'] / df['total_nights'].replace(0, 1)
    df['has_previous_cancel'] = (df['previous_cancellations'] > 0).astype(int)
    df['is_repeated_guest']   = df['is_repeated_guest'].fillna(0)
    df = df[(df['total_guests'] > 0) & (df['total_nights'] > 0)]

    X = df[features].fillna(0).values
    y = df['is_canceled'].values

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    model = build_model(X_train_s.shape[1])
    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]
    print("🚀 Training …")
    model.fit(X_train_s, y_train, validation_data=(X_val_s, y_val),
              epochs=60, batch_size=256, callbacks=cb_list, verbose=1)

    y_prob = model.predict(X_test_s, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    accuracy = float(accuracy_score(y_test, y_pred))
    auc      = float(roc_auc_score(y_test, y_prob))
    print(f"✅ Accuracy: {accuracy*100:.2f}%  AUC: {auc:.4f}")

# ── Routes ───────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           accuracy=round(accuracy * 100, 2),
                           auc=round(auc, 4))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        lead_time          = float(data.get("lead_time", 0))
        total_guests       = float(data.get("total_guests", 2))
        total_nights       = float(data.get("total_nights", 3))
        prev_cancel        = float(data.get("prev_cancel", 0))
        prev_not_canceled  = float(data.get("prev_not_canceled", 0))
        booking_changes    = float(data.get("booking_changes", 0))
        car_parking        = float(data.get("car_parking", 0))
        special_requests   = float(data.get("special_requests", 0))
        adr_val            = float(data.get("adr", 100))
        is_repeated        = float(data.get("is_repeated", 0))

        revenue_est   = adr_val * max(total_nights, 1)
        guest_p_night = total_guests / max(total_nights, 1)
        has_prev      = int(prev_cancel > 0)

        row = np.array([[
            lead_time, total_guests, total_nights,
            prev_cancel, prev_not_canceled,
            booking_changes, car_parking, special_requests,
            adr_val, revenue_est, guest_p_night,
            has_prev, is_repeated, 0
        ]])
        row_s = scaler.transform(row)
        prob  = float(model.predict(row_s, verbose=0)[0][0])

        if prob >= 0.65:
            risk, color = "HIGH", "red"
        elif prob >= 0.40:
            risk, color = "MEDIUM", "amber"
        else:
            risk, color = "LOW", "green"

        return jsonify({
            "risk": risk,
            "color": color,
            "cancel_prob": round(prob * 100, 1),
            "stay_prob":   round((1 - prob) * 100, 1),
            "accuracy":    round(accuracy * 100, 2),
            "auc":         round(auc, 4),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

if __name__ == "__main__":
    train()
    app.run(debug=False, host="0.0.0.0", port=5000)
