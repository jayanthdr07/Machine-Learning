import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Fixed file path string and added raw string prefix
data = pd.read_csv(r"C:\Users\20241CSE0404\Downloads\Salary_Data.csv")

# Fixed missing quotation marks for print and display statements
print("First Five records")
display(data.head())

print("\nDataset Information")
data.info()

print("\nStatistical Summary")
display(data.describe())

# Fixed column name strings
x = data[["YearsExperience"]]
y = data["Salary"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Fixed print statement strings and commas
print("\n--- Model Performance ---")
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R-squared Score (R2):", r2_score(y_test, y_pred))

# Fixed color string names ('blue', 'green', 'red')
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color="blue", label="Training Data")
plt.scatter(x_test, y_test, color="green", label="Test Data")
plt.plot(
    x_train, model.predict(x_train), color="red", linewidth=2, label="Regression Line"
)
plt.title("Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()