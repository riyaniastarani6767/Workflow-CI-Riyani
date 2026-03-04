

import pandas as pd
import mlflow
import mlflow.sklearn
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Bank Marketing Experiment")

df = pd.read_csv("dataset_preprocessing/bank_marketing_clean.csv")

X = df.drop("deposit", axis=1)
y = df["deposit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    n_jobs=-1
)


with mlflow.start_run():

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)


    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("best_n_estimators", grid.best_params_["n_estimators"])
    mlflow.log_param("best_max_depth", grid.best_params_["max_depth"])

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("training_confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("training_confusion_matrix.png")


    metric_info = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    with open("metric_info.json", "w") as f:
        json.dump(metric_info, f)

    mlflow.log_artifact("metric_info.json")


    estimator_html = f"""
    <html>
    <body>
    <h2>Best Model</h2>
    <p>Model: RandomForestClassifier</p>
    <p>n_estimators: {grid.best_params_['n_estimators']}</p>
    <p>max_depth: {grid.best_params_['max_depth']}</p>
    </body>
    </html>
    """

    with open("estimator.html", "w") as f:
        f.write(estimator_html)

    mlflow.log_artifact("estimator.html")

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model"
    )


    print("Best Parameters:", grid.best_params_)
    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)