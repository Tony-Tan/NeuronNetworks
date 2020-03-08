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

    def process(self, return_trajectory=False, hessian_matrix=None):
        trajectory = []
        if return_trajectory:
            trajectory.append(self.x_temp)
        while not self.terminate_test():
            if hessian_matrix is not None:
                self.mininmizing_alone_a_line(hessian_matrix)
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

    def mininmizing_alone_a_line(self,A):
        p = -self.gradient_value_current.transpose()
        self.lr=-(self.gradient_value_current.dot(p))/(p.transpose().dot(A).dot(p))

    def terminate_test(self, threshold_zero=0.001):
        function_value_delta = np.abs(self.function(self.x_temp)-
                                      self.function(self.lr*self.gradient_value_current))
        if function_value_delta >= threshold_zero:
            return False
        return True


def function(x):
    return (x[0]**2+x[1]**2+x[0]*x[1])

def function_g(x):
    return np.array([2*x[0]+x[1], x[0]+2*x[1]])


if __name__ == '__main__':
    n = 400
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)

    X, Y = np.meshgrid(x, y)
    x_0 = np.array([0.8,-0.25])
    lr = 0.041
    sd = SteepestDescent(function, function_g, x_0, lr)
    flag, trajectory = sd.process(return_trajectory=True,hessian_matrix=np.array([[2,1],[1,2]]))
    plt.figure(figsize=(8, 6))
    Z=[0.02,0.06,0.1,0.14,0.18,0.4,0.8,1.2,1.6,2.0,4,6,8,10,12]
    C = plt.contour(X, Y, function([X, Y]), Z, alpha=0.7, cmap=plt.cm.hot)
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
        plt.savefig('contour_plot_' + str(loop_i) + '.png')
        loop_i = loop_i+1
    plt.show()