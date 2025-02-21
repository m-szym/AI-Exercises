import numpy as np
import matplotlib.pyplot as plt


def plot_regression_line(x, y, solution):
    line_x = np.linspace(min(x), max(x), 100)
    line_y = float(solution[0]) + float(solution[1]) * line_x
    plt.plot(line_x, line_y)
    plt.scatter(x, y)
    plt.xlabel('Weight')
    plt.ylabel('MPG')
    plt.show()


def z_standardize(values, population):
    return (values - np.mean(population)) / np.std(population)


def get_observation_matrix(data):
    return np.column_stack((np.ones(data.shape[0]), data))


def calculate_closed_form_solution(observation, y):
    return np.linalg.inv(observation.T.dot(observation)).dot(observation.T).dot(y)


def mse(theta, x, y):
    return np.sum((theta.dot(x.T) - y) ** 2) / x.shape[0]


def mse_gradient(observation, theta, proper_output):
    return (2 / observation.shape[0]) * observation.T.dot((observation.dot(theta) - proper_output))


def gradient_descent(theta, learning_rate, gradient):
    return theta - learning_rate * gradient


def batch_gradient_descent(x, y, learning_rate=0.1, epochs=1000):
    rng = np.random.default_rng()
    theta = rng.random((2,))
    observation = get_observation_matrix(x)

    for e in range(epochs):
        grad = mse_gradient(observation, theta, y)
        theta = gradient_descent(theta, learning_rate, grad)

    return theta


def batch_gradient_descent_with_shuffle(x, y, lre=0.1, epochs=100, batch_size=30, progress=0):
    rng = np.random.default_rng()
    theta = rng.random((2,))
    obs = get_observation_matrix(x)

    xy = np.column_stack((obs, y))

    for e in range(epochs):
        rng.shuffle(xy)

        for batch in np.array_split(xy, [*range(batch_size, xy.shape[0], batch_size)]):
            e_x = batch[:, 0:2]
            e_y = batch[:, 2]
            theta = gradient_descent(theta, lre, mse_gradient(e_x, theta, e_y))

        if progress > 0 and e % progress == 0:
            line_x = np.linspace(min(x), max(x), 100)
            line_y = float(theta[0]) + float(theta[1]) * line_x
            plt.plot(line_x, line_y)

    return theta

