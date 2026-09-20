from sklearn.linear_model import LogisticRegression
import joblib

# Simple training data
X = [
    [0.1, 0.2],
    [0.2, 0.3],
    [0.3, 0.4],
    [0.7, 0.6],
    [0.8, 0.7],
    [0.9, 0.8]
]

y = [0, 0, 0, 1, 1, 1]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save trained model
joblib.dump(model, "model.pkl")

print("Model trained and saved as model.pkl")
