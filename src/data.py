import pandas as pd
from numpy.random import PCG64, Generator

# NOTE: This is a legacy method and should be avoided
# np.random.seed(42)

# number of rows

# n = 10_000

# random data frame creation for age, number of emergency rooms visits, consultations, hospitalizations and exams

# df = pd.DataFrame(
#     {
#         "age": np.random.randint(18, 85, n),
#         "er_visits_12m": np.random.poisson(1.5, n),
#         "consultations_12m": np.random.poisson(5, n),
#         "hospitalizations_12m": np.random.poisson(0.3, n),
#         "exams_12m": np.random.poisson(8, n),
#     }
# )

# A more modern approach to create the data above is the following:


def create_beneficiary_data(n: int, high_quantile: float = 0.9) -> pd.DataFrame:
    """
    This function returns a pandas DataFrame containing randomly
    generated data for health care beneficiaries.

    Parameters:
    -----------
    n: int
      the number of samples (beneficiaries)
    high_quantile: float
      the quantile defining high cost
      Default is 0.9

    Returns
    -------
    df: pd.DataFrame
      a data frame containing the random samples.
    """
    rng = Generator(PCG64())  # use numpy.random.Generator
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 85, n),
            "er_visits_12m": rng.poisson(1.5, n),
            "consultations_12m": rng.poisson(5, n),
            "hospitalizations_12m": rng.poisson(0.3, n),
            "exams_12m": rng.poisson(8, n),
        }
    )

    df["historical_cost"] = (
        500
        + df["age"] * 30
        + df["er_visits_12m"] * 700
        + df["consultations_12m"] * 100
        + df["hospitalizations_12m"] * 5000
        + df["exams_12m"] * 80
        + rng.normal(0, 1000, n)
    ).clip(lower=0)

    # and also some future cost

    df["future_cost"] = (
        600
        + df["age"] * 35
        + df["er_visits_12m"] * 900
        + df["consultations_12m"] * 120
        + df["hospitalizations_12m"] * 6000
        + df["exams_12m"] * 100
        + rng.normal(0, 1500, n)
    ).clip(lower=0)

    p90 = df["future_cost"].quantile(high_quantile)
    df["high_cost"] = (df["future_cost"] > p90).astype(int)

    return df
