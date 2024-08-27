import numpy as np
import matplotlib.pyplot as plt

# Line parameters
real_slope = 2
real_intercept = 5

# Creating the dataset
def create_data(n, real_slope, real_intercept, fault_std):
    x = np.random.uniform(0, 10, n)
    faults = np.random.normal(0, fault_std, n)
    y = real_slope * x + real_intercept + faults
    return x, y

x, y = create_data(100, real_slope, real_intercept, fault_std=2)

# Plotting data
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


class LineModel:
    def __init__(self):
        self.slope = np.random.randn()
        self.intersecting_point = np.random.randn()

    def guess_it(self, x):
        return self.slope * x + self.intersecting_point

    def gradient(self, x, y):
        guess = self.guess_it(x)
        slope_gradient = 2 * np.mean((guess - y) * x)
        intersecting_point_gradients = 2 * np.mean(guess - y)
        return slope_gradient, intersecting_point_gradients

    def update(self, slope_gradient, intersecting_point_gradients, learning_speed):
        self.slope -= learning_speed * slope_gradient
        self.intersecting_point -= learning_speed * intersecting_point_gradients


# Training function
def train(model, x, y, learning_speed, epoch_number):
    for epoch in range(epoch_number):
        for i in range(len(x)):
            slope_gradient, intersecting_point_gradients = model.gradient(x[i], y[i])
            model.update(slope_gradient, intersecting_point_gradients, learning_speed)
        if epoch % 100 == 0:
            loss = np.mean((model.guess_it(x) - y) ** 2)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")


# Train model
model = LineModel()
train(model, x, y, learning_speed=0.001, epoch_number=1001)

# Line of the trained model
x_guess = np.linspace(0, 10, 100)
y_guess = model.guess_it(x_guess)

# Plotting data and prediction
plt.scatter(x, y)
plt.plot(x_guess, y_guess, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
