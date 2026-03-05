# import pandas as pd
# import mlflow
# import mlflow.sklearn

# # WAJIB supaya MLflow connect ke UI
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("Bank Marketing Experiment")

# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# # load dataset
# df = pd.read_csv("dataset_preprocessing/bank_marketing_clean.csv")

# X = df.drop("deposit", axis=1)
# y = df["deposit"]

# # split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# with mlflow.start_run():

#     model = RandomForestClassifier(random_state=42)

#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)

#     acc = accuracy_score(y_test, y_pred)

#     mlflow.log_param("model", "RandomForest")
#     mlflow.log_metric("accuracy", acc)

#     mlflow.sklearn.log_model(model, "model")

#     print("Accuracy:", acc)

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# gunakan autolog
mlflow.sklearn.autolog()

# load dataset
data = pd.read_csv("Membangun_model/dataset_preprocessing/bank_marketing_clean.csv")

# pisahkan fitur dan target
X = data.drop("deposit", axis=1)
y = data["deposit"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# mulai MLflow run
with mlflow.start_run():

    # buat model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    # training
    model.fit(X_train, y_train)

    # prediksi
    y_pred = model.predict(X_test)

    # hitung metric
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)