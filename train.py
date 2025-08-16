import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
import tensorflow as tf

data = Path("sign_data.csv")

df = pd.read_csv(data)

# Select only the first 63 columns (21 hand landmarks × 3 coordinates)
X = df.iloc[:, 1:64].values

# Labels
y = df["label"].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)


model = Sequential([
    Dense(256, activation='relu', input_shape=(63,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(0.0001), metrics=["accuracy"])

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

fitting = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=16,
    epochs=500,
    shuffle=True,
    callbacks=[early_stop],
    verbose=2
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy*100:.2f}%")

pkl_filename = "model.pkl"

with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)