import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("bank-marketing-model")

data = pd.read_csv("dataset_preprocessing/bank_marketing_clean.csv")

X = data.drop("deposit", axis=1)
y = data["deposit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.sklearn.autolog()

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")