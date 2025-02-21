import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

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
observation = np.column_stack((np.ones(len(x_train)), x_train))
theta_best = np.linalg.inv(observation.T.dot(observation)).dot(observation.T).dot(y_train)
print(f"Closed-form solution: {theta_best}")

# TODO: calculate error
closed_form_error = (2 / observation.shape[0]) * observation.T.dot((observation.dot(theta_best) - y_train))
print(f"MSE of closed for closed-form solution: {closed_form_error}")

# plot the regression line
# x = np.linspace(min(x_test), max(x_test), 100)
# y = float(theta_best[0]) + float(theta_best[1]) * x
# plt.plot(x, y)
# plt.scatter(x_test, y_test)
# plt.xlabel('Weight')
# plt.ylabel('MPG')
# plt.show()


# TODO: standardization
def standardize(data, population):
    return (data - np.mean(population)) / np.std(population)


x_pop = np.concatenate((x_train, x_test))
x_z = (x_train - np.mean(x_pop)) / np.std(x_pop)

y_pop = np.concatenate((y_train, y_test))
y_z = (y_train - np.mean(y_pop)) / np.std(y_pop)
y_train_std = standardize(y_train, np.concatenate((y_train, y_test)))

xzt = (x_test - np.mean(x_train)) / np.std(x_train)
yzt = (y_test - np.mean(y_train)) / np.std(y_train)

# TODO: calculate theta using Batch Gradient Descent
rng = np.random.default_rng()
learning_rate = 0.1
epochs = 100
batch_size = 30

xy = np.column_stack((np.ones(x_z.shape[0]), x_z, y_z))
obs = np.column_stack((np.ones(len(x_z)), x_z))

b_theta = rng.random((2,))
for e in range(epochs):
    # rng.shuffle(xy)

    b_theta = b_theta - learning_rate * (2 / obs.shape[0]) * obs.T.dot((obs.dot(b_theta) - y_z))

    for batch in np.array_split(xy, [*range(batch_size, xy.shape[0], batch_size)]):
        e_x = batch[:, 0:2]
        e_y = batch[:, 2]
        b_theta = b_theta - learning_rate * (2 / e_x.shape[0]) * e_x.T.dot((e_x.dot(b_theta) - e_y))

    if e % 10 == 0:
        x = np.linspace(min(xzt), max(xzt), 100)
        y = float(b_theta[0]) + float(b_theta[1]) * x
        plt.plot(x, y)

# TODO: calculate error

# plot the regression line
x = np.linspace(min(xzt), max(xzt), 100)
y = float(b_theta[0]) + float(b_theta[1]) * x
plt.plot(x, y, color="black")
plt.scatter(xzt, yzt)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
