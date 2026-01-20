import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/eco_driving_score.csv")

# Convert eco_score to classes
def eco_category(score):
    if score < 40:
        return "Low"
    elif score <= 60:
        return "Medium"
    else:
        return "High"

df["eco_class"] = df["eco_score"].apply(eco_category)

# Features and target
X = df.drop(["eco_score", "eco_class"], axis=1)
y = df["eco_class"]

le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save PKL files (small size)
joblib.dump(model, "models/eco_model_small.pkl", compress=3)
joblib.dump(le, "models/label_encoder.pkl", compress=3)
joblib.dump(X.columns.tolist(), "models/model_features.pkl", compress=3)

print("✅ Model training completed and PKL files saved!")
