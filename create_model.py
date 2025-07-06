import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

np.random.seed(42)
n_samples = 500

data = {
    'speed': np.random.uniform(40, 60, n_samples),
    'stamina': np.random.uniform(30, 50, n_samples),
    'jockey_rating': np.random.uniform(1, 10, n_samples),
    'track_condition': np.random.choice([0, 1, 2], n_samples),
}

df = pd.DataFrame(data)
df['winner'] = (
    (df['speed'] * 0.4 + df['stamina'] * 0.3 + df['jockey_rating'] * 0.2 - df['track_condition'] * 5)
    + np.random.normal(0, 5, n_samples)
)
df['winner'] = pd.qcut(df['winner'], q=3, labels=[0, 1, 2])

X = df[['speed', 'stamina', 'jockey_rating', 'track_condition']]
y = df['winner'].astype(int)

model = RandomForestClassifier()
model.fit(X, y)

os.makedirs('model', exist_ok=True)
with open('model/horse_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to model/horse_model.pkl")
