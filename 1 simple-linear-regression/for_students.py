from data import get_data, inspect_data, split_data

from functions import *

data = get_data()
# inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
observation = get_observation_matrix(x_train)
theta_best = calculate_closed_form_solution(observation, y_train)
print(f"Closed-form solution: {theta_best}")

# TODO: calculate error
closed_form_error = mse(theta_best, get_observation_matrix(x_test), y_test)
print(f"MSE of closed for closed-form solution: {closed_form_error}")


# plot the regression line
plot_regression_line(x_test, y_test, theta_best)


# TODO: standardization
x_train_std = z_standardize(x_train, x_train)
y_train_std = z_standardize(y_train, y_train)
x_test_std = z_standardize(x_test, x_train)
y_test_std = z_standardize(y_test, y_train)

# TODO: calculate theta using Batch Gradient Descent
d_theta = batch_gradient_descent(x_train_std, y_train_std, epochs=100)
print(f"Batch gradient descent standardized solution: {d_theta}")

# TODO: calculate error
d_mse = mse(d_theta, get_observation_matrix(x_test_std), y_test_std)
print(f"MSE of batch gradient descent standardized solution: {d_mse}")

# plot the regression line
plot_regression_line(x_test_std, y_test_std, d_theta)

