from fastapi import FastAPI
import pickle

app = FastAPI()

with open("pipeline_v1.bin", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(client: dict):
    X = [client]
    prob = model.predict_proba(X)[0, 1]
    return {"probability": prob}
