import time
import joblib
import numpy as np
from prometheus_client import start_http_server, Counter, Histogram

# load model
model = joblib.load("model.pkl")

# prometheus metrics
prediction_requests = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

prediction_errors = Counter(
    "prediction_errors_total",
    "Total number of prediction errors"
)

prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds"
)

# prediction function with metrics
def predict(data):
    prediction_requests.inc()

    start_time = time.time()

    try:
        data = np.array(data).reshape(1, -1)
        result = model.predict(data)

    except Exception:
        prediction_errors.inc()
        raise

    latency = time.time() - start_time
    prediction_latency.observe(latency)

    return result


# run the Prometheus exporter
if __name__ == "__main__":
    print("Starting Prometheus exporter on port 8000...")
    start_http_server(8000)

    while True:
        try:
            
            sample_data = [1, 2, 3, 4]
            prediction = predict(sample_data)

            print("Prediction:", prediction)

        except Exception as e:
            print("Prediction error:", e)

        time.sleep(5)