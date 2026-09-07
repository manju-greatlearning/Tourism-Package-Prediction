import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model():
    X_test = pd.read_csv("artifacts/test_X.csv")
    y_test = pd.read_csv("artifacts/test_y.csv").values.ravel()
    model = joblib.load("models/model.pkl")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred))

if __name__ == "__main__":
    evaluate_model()
