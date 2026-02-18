# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Step 1: Create Dataset (Example: Study Hours vs Marks)
X = np.array([[10], [20], [30], [40], [50], [60]])

y = np.array([35, 45, 55, 65, 75, 85])

# Step 2: Create Random Forest Regressor model
RandomForestRegModel = RandomForestRegressor(n_estimators=100, random_state=0)

# Step 3: Train the model
RandomForestRegModel.fit(X, y)

# Step 4: Predict for new value
X_marks = np.array([[70]])
prediction = RandomForestRegModel.predict(X_marks)

print("Predicted Marks for 70 hours of study:", prediction[0])

# Step 5: Visualize the results
X_grid = np.arange(min(X), max(X)+20, 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red')
plt.plot(X_grid, RandomForestRegModel.predict(X_grid), color='blue')
plt.title('Random Forest Regression')
plt.xlabel('Study Hours')
plt.ylabel('Marks')
plt.show()
