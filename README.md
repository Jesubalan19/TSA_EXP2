# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
# Date:-08-2025
### NAME:JESUBALAN A
### REGISTER NO:212223240060
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
### A - LINEAR TREND ESTIMATION
```PYTHON
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

file_path = "/mnt/data/TSLA.csv"
data = pd.read_csv("TSLA.csv")

data["Date"] = pd.to_datetime(data["Date"])
data["Year"] = data["Date"].dt.year

yearly_tsla = data.groupby("Year")["Close"].mean().reset_index()

X = yearly_tsla["Year"].values.reshape(-1, 1)
y = yearly_tsla["Close"].values

linear_model = LinearRegression()
linear_model.fit(X, y)

yearly_tsla["Linear_Trend"] = linear_model.predict(X)

plt.figure(figsize=(10,6))
plt.plot(yearly_tsla["Year"], y, label="Average Close Price", marker="o")
plt.plot(yearly_tsla["Year"], yearly_tsla["Linear_Trend"], color="orange", label="Linear Trend")
plt.title("Linear Trend Estimation - TSLA Stock Prices")
plt.xlabel("Year")
plt.ylabel("Average Close Price")
plt.legend()
plt.grid(True)
plt.show()
```
### B- POLYNOMIAL TREND ESTIMATION
```PYTHON
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

yearly_tsla["Poly_Trend"] = poly_model.predict(X_poly)

plt.figure(figsize=(10,6))
plt.plot(yearly_tsla["Year"], y, label="Average Close Price", marker="o", alpha=0.6)
plt.plot(yearly_tsla["Year"], yearly_tsla["Poly_Trend"], color="red", label="Polynomial Trend (Degree 2)")
plt.title("Polynomial Trend Estimation - TSLA Stock Prices")
plt.xlabel("Year")
plt.ylabel("Average Close Price")
plt.legend()
plt.grid(True)
plt.show()
```
### OUTPUT
### A - LINEAR TREND ESTIMATION
<img width="1149" height="683" alt="Screenshot 2025-08-26 141351" src="https://github.com/user-attachments/assets/71cc560e-ec7b-45c3-a89c-8936e170432d" />




### B- POLYNOMIAL TREND ESTIMATION
<img width="1184" height="683" alt="Screenshot 2025-08-26 141322" src="https://github.com/user-attachments/assets/deac1e66-deae-4f01-b401-eea1e7b1104b" />


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
