# Import libraries
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import statsmodels.api as sm

# Read dataset
data = pd.read_csv("Moisture Data.csv")

# Print the data
print(data)

# Plot the scatter plot 
plt.scatter(data.moisture, data.viscosity)
plt.xlabel('Moisture')
plt.ylabel('Viscosity')
plt.title('Moisture vs. Viscosity')
plt.show()

# Mean square error function
def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].moisture
        y = points.iloc[i].viscosity
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))  # Return the MSE

# Mean Gradient descent function
def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)
    
    for i in range(n):
        x = points.iloc[i].moisture
        y = points.iloc[i].viscosity
        
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
        
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

# Initialize parameters
m = 0 # Initial slope of the regression line
b = 0 # Initial intercept of the regression line
L = 0.001 # Learning rate, affecting the size of each step in gradient descent
epochs = 300 # Number of iterations for the gradient descent algorithm

# Perform gradient descent
for i in range(epochs):
    m, b = gradient_descent(m, b, data, L)

print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")

# Plot the data and the regression line
plt.scatter(data.moisture, data.viscosity, color="blue", label="Data Points")
x_values = np.linspace(data.moisture.min(), data.moisture.max(), 100)
y_values = m * x_values + b
plt.plot(x_values, y_values, color="red", label="Regression Line")
plt.xlabel('Moisture')
plt.ylabel('Viscosity')
plt.title('Moisture vs. Viscosity with Regression Line')
plt.legend()
plt.show()

# Compute the Mean Squared Error
mse = loss_function(m, b, data)
print(f"Mean Squared Error: {mse}")

# Compute R-squared
def r_squared(points, m, b):
    y_mean = points.viscosity.mean()
    total_variance = sum((points.viscosity - y_mean) ** 2)
    explained_variance = sum((m * points.moisture + b - y_mean) ** 2)
    return explained_variance / total_variance

r2 = r_squared(data, m, b)
print(f"R-squared: {r2}")

data['intercept'] = 1


# Perform linear regression using statsmodels
model = sm.OLS(data.viscosity, data[['intercept', 'moisture']])
results = model.fit()

# Print the summary of the regression
print(results.summary())

# Extract the t-statistic and p-value for the slope
t_stat = results.tvalues['moisture']
p_value = results.pvalues['moisture']

print(f"T-statistic for the slope: {t_stat}")
print(f"P-value for the slope: {p_value}")
