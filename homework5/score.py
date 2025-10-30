import pickle

# Load the model
with open("pipeline_v1.bin", "rb") as f:
    model = pickle.load(f)

# Define the client
client = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# Predict probability
X = [client]
prob = model.predict_proba(X)[0, 1]
print(prob)
