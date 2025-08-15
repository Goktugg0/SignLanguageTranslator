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

print(y)