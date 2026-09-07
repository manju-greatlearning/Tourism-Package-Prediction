import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_model():
    os.makedirs("models", exist_ok=True)  # Ensure models folder exists

    X_train = pd.read_csv("artifacts/train_X.csv")
    y_train = pd.read_csv("artifacts/train_y.csv").values.ravel()

    rf = RandomForestClassifier(random_state=42)
    param_grid = {"n_estimators":[100,200], "max_depth":[5,10,None]}
    grid = GridSearchCV(rf, param_grid, cv=3, scoring="f1", verbose=1)
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    best_model = grid.best_estimator_

    joblib.dump(best_model, "models/model.pkl")
    print("Model saved to models/model.pkl")

if __name__ == "__main__":
    train_model()
