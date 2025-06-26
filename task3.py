import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = {
    "Player": ["Virat Kohli", "Rohit Sharma", "KL Rahul", "Shubman Gill", "Rishabh Pant",
               "Hardik Pandya", "Ravindra Jadeja", "Jasprit Bumrah", "Mohammed Siraj", "Kuldeep Yadav"],
    "Age": [35, 36, 32, 25, 27, 30, 36, 31, 30, 29],
    "Salary": [57.3, 49.2, 44.1, 48.5, 41.2, 34.7, 32.5, 12.1, 10.4, 14.6]
}

df = pd.DataFrame(data)
df.to_csv("indian_cricketers.csv", index=False)

df = pd.read_csv("indian_cricketers.csv")

X = df[["Age"]]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel("Age")
plt.ylabel("Salary (in Lakhs)")
plt.title("Simple Linear Regression: Salary vs Age")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
