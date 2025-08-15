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

# Features = all columns except "label"
X = df.drop("label", axis=1).values
# Labels
y = df["label"].values

# Initialize and fit LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))



model = Sequential([
    Dense(16, activation='relu', input_shape=(189,)),
    Dense(32, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])


model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(0.00001), metrics=["accuracy"])


fitting = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=16,
    epochs=30,
    shuffle=True,
    verbose=2,
)