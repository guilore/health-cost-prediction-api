# Health Cost Prediction API

End-to-end Machine Learning project for predicting whether a healthcare beneficiary is likely to become a **high-cost beneficiary in a future period**.

The project demonstrates the complete path from data preparation and model development to model serving through a REST API using **FastAPI**.

> **Note:** This project uses synthetically generated healthcare data for educational and portfolio purposes. The dataset does not represent real patients or real healthcare claims.

---

## Business Problem

Healthcare organizations and health-plan providers need to identify beneficiaries who may generate unusually high healthcare costs in the future.

Early identification of high-cost beneficiaries can support initiatives such as:

- Preventive and care-management programs
- Targeted interventions
- Resource allocation
- Cost forecasting
- Healthcare utilization management

The objective of this project is to build a classification model capable of estimating the probability that a beneficiary will belong to the **highest-cost group in the following period**.

---

## Project Objective

Given historical demographic, healthcare utilization, and cost information, predict whether a beneficiary will be classified as **high cost in a future period**.

The target variable is defined using the 90th percentile of future healthcare costs:

```text
high_cost = 1 → future cost > 90th percentile
high_cost = 0 → otherwise
```

This creates an imbalanced classification problem in which approximately **10% of beneficiaries belong to the positive class**.

---

## Dataset

The dataset contains **10,000 synthetic beneficiaries**.

### Features

| Feature | Description |
|---|---|
| `age` | Beneficiary age |
| `er_visits_12m` | Emergency room visits during the previous 12 months |
| `consultations_12m` | Medical consultations during the previous 12 months |
| `hospitalizations_12m` | Hospitalizations during the previous 12 months |
| `exams_12m` | Medical exams during the previous 12 months |
| `historical_cost` | Historical healthcare cost |

### Target

| Variable | Description |
|---|---|
| `future_cost` | Healthcare cost in the future period |
| `high_cost` | Binary target indicating whether future cost is above the 90th percentile |

The model only uses information available from the historical period to predict the future target, helping avoid target leakage.

---

## Machine Learning Approach

### 1. Data Preparation

The synthetic dataset was generated using demographic, healthcare utilization, and historical cost variables.

The target was created from future healthcare costs rather than from the historical utilization variables.

### 2. Train/Test Split

The dataset was divided into:

- **80% training**
- **20% testing**

Stratified sampling was used to preserve the proportion of high-cost beneficiaries in both datasets.

### 3. Model

The project uses **LightGBM**, a gradient boosting framework based on decision trees.

The model was configured with:

```python
LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)
```

LightGBM was selected because tree-based gradient boosting models are effective for tabular data and can capture nonlinear relationships and interactions between healthcare utilization variables.

---

## Model Evaluation

The model was evaluated using metrics appropriate for an imbalanced classification problem.

### Results

| Metric | Result |
|---|---:|
| ROC-AUC | **0.975** |
| PR-AUC | **0.844** |
| Precision — High Cost | **0.78** |
| Recall — High Cost | **0.67** |
| F1 Score — High Cost | **0.72** |

At the default classification threshold of 0.50, the model identified 67% of the high-cost beneficiaries in the test set, with a precision of 78%.

### Why PR-AUC?

Because the high-cost class represents only approximately 10% of the population, accuracy alone would not provide a meaningful evaluation of model performance.

Precision-Recall metrics provide a better view of how effectively the model identifies the minority class.

---

## Model Serving

After training, the model is serialized using `joblib` and served through a **FastAPI REST API**.

The API exposes a `/predict` endpoint that receives beneficiary information and returns:

- Probability of being classified as high cost
- Binary high-cost prediction

### Example Request

```json
{
    "age": 72,
    "er_visits_12m": 5,
    "consultations_12m": 8,
    "hospitalizations_12m": 1,
    "exams_12m": 15,
    "historical_cost": 18000
}
```

### Example Response

```json
{
    "probabilidade": 0.XX,
    "high_cost": 1
}
```

The API can also be explored interactively through FastAPI's automatically generated Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Project Structure

```text
health-cost-prediction-api/
│
├── health-cost-api.ipynb   # Data generation, exploration, training and evaluation
├── app.py                  # FastAPI application
├── model.joblib            # Trained LightGBM model
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- Joblib
- FastAPI
- Uvicorn
- Git
- GitHub

---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/guilore/health-cost-prediction-api.git
cd health-cost-prediction-api
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API

```bash
python -m uvicorn app:app --reload
```

### 5. Open the API documentation

Navigate to:

```text
http://127.0.0.1:8000/docs
```

From there, the `/predict` endpoint can be tested directly through the Swagger interface.

---

## Key Data Science Considerations

This project was designed to demonstrate several practical Machine Learning concepts:

- Imbalanced classification
- Stratified train/test splitting
- ROC-AUC and PR-AUC evaluation
- Precision/Recall trade-offs
- Probability-based predictions
- Classification threshold selection
- Target leakage prevention
- Gradient boosting for tabular data
- Model serialization
- REST API model serving

---

## Limitations

This project uses **synthetically generated data**. Therefore, the reported model performance should not be interpreted as representative of real-world healthcare prediction performance.

In a production healthcare environment, additional considerations would be required, including:

- Real claims and utilization data
- Temporal validation
- Missing data treatment
- Feature engineering
- Model calibration
- Bias and fairness assessment
- Explainability
- Data privacy and security
- Model monitoring
- Data drift and concept drift
- Clinical and business validation

---

## Future Improvements

Planned extensions include:

- [ ] Hyperparameter optimization
- [ ] Cross-validation
- [ ] SHAP-based model explainability
- [ ] Probability calibration
- [ ] Automated testing
- [ ] Docker containerization
- [ ] Cloud deployment
- [ ] Model monitoring
- [ ] CI/CD pipeline
- [ ] Production-oriented logging

---

## Author

**Guilherme Lorenzen**

Data Scientist / Data Analyst with experience in statistical analysis, Machine Learning, healthcare analytics, SQL, Python, and Business Intelligence.

This project was developed as part of a hands-on portfolio focused on **Machine Learning, API development, and production-oriented data science workflows**.
