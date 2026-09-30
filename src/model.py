# Here we add the ML model

from collections.abc import Sequence

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

FEATURES = [
    "age",
    "er_visits_12m",
    "consultations_12m",
    "hospitalizations_12m",
    "exams_12m",
    "historical_cost",
]

TARGET = "high_cost"


def split_data(data: pd.DataFrame, test_size: float) -> Sequence[pd.DataFrame]:
    """
    This function returns a sequence of pandas DataFrames containing
    the data X and y data split into test and train samples.

    Parameters:
    -----------
    data: pd.DataFrame
      the data frame containing the data
    test_size: float
      the size of the test sample

    Returns
    -------
    X_train, X_test, y_train, y_test: pd.DataFrame
      a sequence of data frames containing the test and train data.
    """

    try:
        X = data[FEATURES]
    except KeyError as e:
        print("Data is missing some features")
        print(e)

    try:
        y = data[TARGET]
    except KeyError as e:
        print("Data is missing some features")
        print(e)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        # random_state=42,
        stratify=y,  # stratify because the target is unbalanced
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> LGBMClassifier:
    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        verbosity=-1,
    )

    model.fit(X_train, y_train)

    return model


def predict(model: LGBMClassifier, X_test: pd.DataFrame):
    return model.predict_proba(X_test)[:, 1]
