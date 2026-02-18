import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Dataset
df = pd.read_csv("NBD.csv")

# Step 2: Define Features (X) and Target (y)
x = df.drop('diabetes', axis=1)   # Independent variables
y = df['diabetes']                # Dependent variable (Target)

# Step 3: Split Data into Training and Testing
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

# Step 4: Create Model
model = GaussianNB()

# Step 5: Train Model
model.fit(x_train, y_train)

# Step 6: Predict
y_pred = model.predict(x_test)

# Step 7: Print Predictions
print("Predicted Values:")
print(y_pred)

# Step 8: Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 9: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
