import pandas as pd

def validate_dataset(path="data/tourism.csv"):
    expected_columns = [
        "CustomerID","ProdTaken","Age","TypeofContact","CityTier","Occupation",
        "Gender","NumberOfPersonVisiting","PreferredPropertyStar","MaritalStatus",
        "NumberOfTrips","Passport","OwnCar","NumberOfChildrenVisiting",
        "Designation","MonthlyIncome","PitchSatisfactionScore","ProductPitched",
        "NumberOfFollowups","DurationOfPitch"
    ]
    df = pd.read_csv(path)
    print("Dataset shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())
    print("Column types:\n", df.dtypes)
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    print("Validation passed.")
    return df

if __name__ == "__main__":
    validate_dataset()
