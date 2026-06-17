import pandas as pd

def load_data():
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race",
        "sex", "capital-gain", "capital-loss", "hours-per-week",
        "native-country", "income"
    ]

    # Load training data
    train_df = pd.read_csv(
        "adult.data",
        names=columns,
        sep=",",
        skipinitialspace=True
    )

    # Load test data
    test_df = pd.read_csv(
        "adult.test",
        names=columns,
        sep=",",
        skiprows=1,  # first row is useless header
        skipinitialspace=True
    )

    # Fix labels (test set has ".")
    test_df["income"] = test_df["income"].str.replace(".", "", regex=False)

    # Combine both
    df = pd.concat([train_df, test_df], ignore_index=True)

    return df