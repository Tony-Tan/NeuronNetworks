import numpy as np
import matplotlib.pyplot as plt


class NewtonMethod():
    def __init__(self, function, gradient, hessian_matrix, initial_point):
        self.x_0 = initial_point
        self.x_temp = self.x_0
        self.function = function
        self.gradient_function = gradient
        self.hessian_matrix = hessian_matrix
        self.gradient_value_current = self.gradient_function(self.x_0)

    def process(self, return_trajectory=False):
        trajectory = []
        if return_trajectory:
            trajectory.append(self.x_temp)
        while not self.terminate_test():
            inv = np.linalg.inv(self.hessian_matrix(self.x_temp))
            g = self.gradient_value_current
            x_current = self.x_temp - inv.dot(g)
            #if self.function(self.x_temp) <= self.function(x_current):
            #    print('Algorithm diverge! stop the process!')
            #    return False, trajectory
            self.x_temp = x_current
            self.gradient_value_current = self.gradient_function(self.x_temp)
            if return_trajectory:
                trajectory.append(self.x_temp)
                print(self.x_temp)
        if return_trajectory:
            return True,trajectory
        return True,None


    def terminate_test(self, threshold_zero=0.001):
        inv = np.linalg.inv(self.hessian_matrix(self.x_temp))
        g = self.gradient_value_current
        function_value_delta = np.linalg.norm(inv.dot(g))
        if function_value_delta >= threshold_zero:
            return False
        print('termination condition achieve')
        return True


def function(x):
    return (x[1]-x[0])**4.+8.*x[0]*x[1]-x[0]+x[1]+3.


def function_g(x):
    return np.array([-4*(x[1]-x[0])**3+8*x[1]-1, 4*(x[1]-x[0])**3+8*x[0]+1])


def function_h(x):
    return np.array([[12*(x[1]-x[0])**2, -12*(x[1]-x[0])**2+8],
                     [-12*(x[1]-x[0])**2+8, 12*(x[1]-x[0])**2]])


def function_app(x,x_org):
    x_1 = x_org[0]
    x_2 = x_org[1]
    H = function_h(np.array([x_1, x_2]))
    g = function_g(np.array([x_1, x_2]))
    x_delta = np.array([[x[0]-x_1], [x[1]-x_2]])
    f1 = function(x_org)
    f2 = g[0]*x_delta[0] + g[1]*x_delta[1]
    f3 = .5*(x_delta[0]**2*H[0][0]+x_delta[0]*x_delta[1]*H[1][0]+x_delta[0]*x_delta[1]*H[0][1]+x_delta[1]**2*H[1][1])
    f =f1 + f2 + f3
    return f[0]


def function_1(x):
    return x[0]**2+25*x[1]**2


def function_1_g(x):
    return np.array([2*x[0],50*x[1]])


def function_1_h(x):
    return np.array([[2, 0],
                     [0, 50]])


if __name__ == '__main__':
    n = 320
    x = np.linspace(-1.6, 1.6, n)
    y = np.linspace(-1.6, 1.6, n)

    X, Y = np.meshgrid(x, y)
    x_0 = np.array([1.15,0.75])
    nm = NewtonMethod(function, function_g, function_h,x_0)
    flag, trajectory = nm.process(return_trajectory=True)
    plt.figure(figsize=(18, 8))
    Z = np.array([1.0,1.4,1.6,
                  2.0,2.4,2.6,
                  3.0,3.4,3.6,
                  4.,6.,8.,10.,14.,18.,22.])
    plt.subplot(1, 2, 1)
    C = plt.contour(X, Y, function([X, Y]), Z, alpha=0.7, cmap=plt.cm.hot)
    plt.clabel(C, inline=True, fontsize=10)
    x_0_last = 0
    x_1_last = 0
    loop_i = 0
    for x in trajectory:
        plt.subplot(1, 2, 1)
        plt.scatter(x[0], x[1], c='b', marker='x', alpha=0.7)

        #
        plt.subplot(1, 2, 2)
        C = plt.contour(X, Y, function_app([X, Y], x), Z, alpha=0.7, cmap=plt.cm.hot)
        plt.clabel(C, inline=True, fontsize=10)
        plt.scatter(x[0], x[1], c='b', marker='x', alpha=0.7)
        if x_0_last != 0 or x_1_last != 0:
            plt.subplot(1, 2, 1)
            plt.plot([x[0], x_0_last], [x[1], x_1_last], c='g', linestyle='-.', linewidth=2, alpha=0.7)
            plt.subplot(1, 2, 2)
            plt.scatter(x_0_last, x_1_last, c='b', marker='x', alpha=0.7)
            plt.plot([x[0], x_0_last], [x[1], x_1_last], c='g', linestyle='-.', linewidth=2, alpha=0.7)
        x_0_last = x[0]
        x_1_last = x[1]
        plt.savefig('contour_plot_' + str(loop_i) + '.png')
        plt.subplot(1, 2, 2)
        plt.cla()
        loop_i = loop_i+1
    plt.show()