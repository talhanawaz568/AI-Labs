import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np

# Create a simple model that 'predicts' based on 4 features
X = np.array([[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]])
y = np.array([0, 1]) # 0: Setosa, 1: Virginica

model = LogisticRegression()
model.fit(X, y)

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✓ 'model.pkl' created successfully.")

