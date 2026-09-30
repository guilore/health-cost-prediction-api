from src.model import split_data, train_model, predict
from src.data import create_beneficiary_data
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

df = create_beneficiary_data(n=1000, high_quantile=0.95)

X_train, X_test, y_train, y_test = split_data(
    data=df,
    test_size=0.3
)

model = train_model(X_train, y_train)

joblib.dump(model, "models/health_cost_model.joblib")

y_prob = predict(model, X_test)

print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("PR-AUC:", average_precision_score(y_test, y_prob))

# Threshold
y_pred = (y_prob >= 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualizing for different thresholds

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

for threshold in thresholds:
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n===== Threshold: {threshold} =====")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

