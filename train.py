from pathlib import Path
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def load_data(path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    return pd.read_csv(Path(path))


def preprocess_data(df: pd.DataFrame):
    """We flatten the array and use scaler and encoder to convert the mixed
       list of datas into encoded values.

       NOTE: If two hands are used the 63 should be 126"""
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

    return X_scaled, y_encoded, le, scaler

def build_model(input_dim: int, num_classes: int):
    """Build a feedforward neural network for classification."""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(0.0001),
        metrics=["accuracy"]
    )
    return model

def train_model(model, x_train, y_train, x_test, y_test):
    """"Train the model couple times until performance starts going down."""
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    fitting = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        batch_size=16,
        epochs=1000,
        shuffle=True,
        callbacks=[early_stop],
        verbose=2
    )
    return fitting

def save_artifacts(model, scaler, encoder):
    """Save model, scaler, and label encoder."""
    with open("scaler.pkl", "wb") as f:
        # noinspection PyTypeChecker
        pickle.dump(scaler, f)

    with open("label_encoder.pkl", "wb") as f:
        # noinspection PyTypeChecker
        pickle.dump(encoder, f)

    model.save("model.keras")

def main():
    df = load_data("sign_data.csv")
    X_scaled, y_encoded, encoder, scaler = preprocess_data(df)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,
                                                        y_encoded, test_size=0.2, random_state=42)
    model = build_model(input_dim=63, num_classes=len(encoder.classes_))

    fitting = train_model(model, X_train, y_train, X_test, y_test)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    save_artifacts(model, scaler, encoder)

if __name__ == "__main__":
    main()