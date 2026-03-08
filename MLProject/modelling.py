# import pandas as pd
# import mlflow
# import mlflow.sklearn

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # gunakan SQLite tracking (CI friendly)
# mlflow.set_tracking_uri("sqlite:///mlflow.db")

# # set experiment
# mlflow.set_experiment("bank-marketing-model")

# # pastikan tidak ada run lama
# mlflow.end_run()

# # load dataset
# data = pd.read_csv("dataset_preprocessing/bank_marketing_clean.csv")

# # split feature dan target
# X = data.drop("deposit", axis=1)
# y = data["deposit"]

# # split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42
# )

# with mlflow.start_run():

#     model = RandomForestClassifier(
#         n_estimators=100,
#         random_state=42
#     )

#     mlflow.log_param("n_estimators", 100)

#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)

#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, average="binary")
#     recall = recall_score(y_test, y_pred, average="binary")
#     f1 = f1_score(y_test, y_pred, average="binary")

#     mlflow.log_metric("accuracy", accuracy)
#     mlflow.log_metric("precision", precision)
#     mlflow.log_metric("recall", recall)
#     mlflow.log_metric("f1_score", f1)

#     mlflow.sklearn.log_model(model, "model")

# print("Training selesai")

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# gunakan sqlite supaya CI tidak error
mlflow.set_tracking_uri("sqlite:///mlruns.db")

mlflow.set_experiment("bank-marketing-model")

# load dataset
data = pd.read_csv("dataset_preprocessing/bank_marketing_clean.csv")

X = data.drop("deposit", axis=1)
y = data["deposit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    mlflow.log_param("n_estimators", 100)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "model")

print("Training selesai")