import numpy as np
import matplotlib.pyplot as plt


class SteepestDescent():
    def __init__(self, function, gradient, initial_point, learning_rate):
        self.lr = learning_rate
        self.x_0 = initial_point
        self.x_temp = self.x_0
        self.function = function
        self.gradient_function = gradient
        self.gradient_value_current = self.gradient_function(self.x_0)

    def process(self, return_trajectory=False):
        trajectory = []
        if return_trajectory:
            trajectory.append(self.x_temp)
        while not self.terminate_test():
            x_current = self.x_temp - self.lr*self.gradient_value_current
            if self.function(self.x_temp) <= self.function(x_current):
                print('Algorithm diverge! stop the process!')
                return False, trajectory
            self.x_temp = x_current
            self.gradient_value_current = self.gradient_function(self.x_temp)
            if return_trajectory:
                trajectory.append(self.x_temp)
                print(self.x_temp)
        if return_trajectory:
            return True,trajectory
        return True,None

    def terminate_test(self, threshold_zero=0.001):
        function_value_delta = np.abs(self.function(self.x_temp)-
                                      self.function(self.lr*self.gradient_value_current))
        if function_value_delta >= threshold_zero:
            return False
        return True


def function(x):
    return (x[0]**2+50*x[1]**2)

def function_g(x):
    return np.array([2*x[0], 50*x[1]])


if __name__ == '__main__':
    n = 256
    x = np.linspace(-0.75, 0.75, n)
    y = np.linspace(-0.75, 0.75, n)

    X, Y = np.meshgrid(x, y)
    x_0 = np.array([0.5,0.5])
    lr = 0.041
    sd = SteepestDescent(function, function_g, x_0, lr)
    flag, trajectory = sd.process(return_trajectory=True)
    plt.figure(figsize=(8, 6))
    C = plt.contour(X, Y, function([X, Y]), 20, alpha=0.7, cmap=plt.cm.hot)
    plt.clabel(C, inline=True, fontsize=10)
    x_0_last = 0
    x_1_last = 0
    loop_i = 0
    for x in trajectory:
        plt.scatter(x[0], x[1], c='b', marker='x', alpha=0.7)
        if x_0_last != 0 and x_1_last != 0:
            plt.plot([x[0], x_0_last], [x[1], x_1_last], c='g', linestyle='-.', linewidth=2, alpha=0.7)
        x_0_last = x[0]
        x_1_last = x[1]
        #plt.savefig('contour_plot_' + str(loop_i) + '.png')
        loop_i = loop_i+1
    plt.show()