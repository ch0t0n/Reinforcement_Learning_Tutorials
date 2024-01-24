import matplotlib.pyplot as plt
import numpy as np


def square_func(x):
    return x**2

def derivative_func(x):
    return 2*x


# Get the x and y for the plot
x_vals = np.arange(-100,100,0.1)
y_vals = square_func(x_vals)

# set the current x and y position
x_curr = 90
y_curr = square_func(x_curr)

# set the learning rate
alpha = 0.01

new_x = x_curr

for _ in range(1000):
    new_x = new_x - alpha * derivative_func(new_x)
    new_y = square_func(new_x)
    plt.plot(x_vals, y_vals)
    plt.plot(new_x, new_y, 'o', color='Red')
    plt.pause(0.1)
    plt.clf()


