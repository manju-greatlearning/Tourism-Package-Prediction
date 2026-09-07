import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(path="data/tourism.csv"):
    # Ensure artifacts directory exists
    os.makedirs("artifacts", exist_ok=True)

    df = pd.read_csv(path)
    df = df.drop(columns=["CustomerID"])  # drop ID

    # Fill missing values
    for col in df.select_dtypes(include="number").columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode categorical
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("ProdTaken", axis=1)
    y = df["ProdTaken"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Save artifacts
    X_train.to_csv("artifacts/train_X.csv", index=False)
    X_test.to_csv("artifacts/test_X.csv", index=False)
    y_train.to_csv("artifacts/train_y.csv", index=False)
    y_test.to_csv("artifacts/test_y.csv", index=False)

    print("Data preparation complete. Artifacts saved.")

if __name__ == "__main__":
    prepare_data()
