# Here we add the ML model

from collections.abc import Sequence
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from src.data import create_beneficiary_data

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


MODEL_PATH = Path("models/health_cost_model.joblib")


def save_model(model: LGBMClassifier) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)


def load_or_train_model() -> LGBMClassifier:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    data = create_beneficiary_data(
        n=1000,
        high_quantile=0.95,
    )

    X_train, X_test, y_train, y_test = split_data(
        data,
        test_size=0.2,
    )

    model = train_model(X_train, y_train)

    save_model(model)

    return model