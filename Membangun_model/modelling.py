import pandas as pd
import mlflow
import mlflow.sklearn

# WAJIB supaya MLflow connect ke UI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Bank Marketing Experiment")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# load dataset
df = pd.read_csv("dataset_preprocessing/bank_marketing_clean.csv")

X = df.drop("deposit", axis=1)
y = df["deposit"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():

    model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", acc)