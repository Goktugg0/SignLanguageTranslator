from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as t

data = Path("sign_data.csv")

df = pd.read_csv(data)

# Features = all columns except "label"
X = df.drop("label", axis=1).values
# Labels
y = df["label"].values

# Initialize and fit LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)