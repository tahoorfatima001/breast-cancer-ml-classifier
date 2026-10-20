"""Helper functions to load the saved model and make predictions."""
import json
import joblib
import pandas as pd

MODEL_PATH = "model/best_model.joblib"
META_PATH = "model/metadata.json"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    meta = json.load(open(META_PATH))
    return model, meta


def predict_one(model, meta, user_values):
    """user_values: dict {feature_name: value} for a subset of features.
    Remaining features are filled with the dataset mean."""
    row = dict(meta["feature_means"])
    row.update(user_values)
    df = pd.DataFrame([row])[meta["feature_names"]]
    proba = model.predict_proba(df)[0]
    # sklearn target: 0 = malignant, 1 = benign
    result = {name: float(p) for name, p in zip(meta["class_names"], proba)}
    label = max(result, key=result.get)
    return label, result


if __name__ == "__main__":
    model, meta = load_artifacts()
    print(predict_one(model, meta, {}))
