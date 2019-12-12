import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def hard_limit(x):
    if x >= 0:
        return 1
    else:
        return 0


def symmetrical_hard_limit(x):
    if x >= 0:
        return 1
    else:
        return -1


def linear(x):
    return x


def saturing_linear(x):
    if x > 1:
        return 1
    elif x >= 0 and x <= 1:
        return x
    else:
        return 0


def symmetric_saturating_linear(x):
    if x >= 1:
        return 1
    elif x > 0 and x < 1:
        return x
    else:
        return -1


def log_sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def hyperbolic_tangent_sigmoid(x):
    return (np.exp(x)-np.exp(-x)) / (np.exp(x)- + np.exp(-x))


def positive_linear(x):
    if x < 0:
        return 0
    else:
        return x


def competitive(x,x_list):

    if x == x_list.max():
        return x
    else:
        return 0


class Peceptron():
    def __init__(self, x, labels, transfer_function_type = 'hard_limit'):
        self.input_bias_padding = np.ones([ np.shape(x)[0],1])
        self.X = np.c_[self.input_bias_padding, x]
        self.labels = labels
        self.weights = np.zeros([1,np.shape(self.X)[1]])
        if transfer_function_type == 'hard_limit':
            self.transfer_function = hard_limit
        elif transfer_function_type == 'symmetrical_hard_limit':
            self.transfer_function = symmetrical_hard_limit
        elif transfer_function_type == 'linear':
            self.transfer_function = linear
        elif transfer_function_type == 'saturing_linear':
            self.transfer_function = saturing_linear
        elif transfer_function_type == 'symmetric_saturating_linear':
            self.transfer_function = symmetric_saturating_linear
        elif transfer_function_type == 'log_sigmoid':
            self.transfer_function = log_sigmoid
        elif transfer_function_type == 'hyperbolic_tangent_sigmoid':
            self.transfer_function = hyperbolic_tangent_sigmoid
        elif transfer_function_type == 'positive_linear':
            self.transfer_function = positive_linear
        else:
            self.transfer_function = hard_limit

    def process(self,x):
        return self.transfer_function(np.dot(self.weights,x.transpose()))

    def convergence_algorithm(self, lr=1.0, max_iteration_num = 1000000):
        for epoch in range(max_iteration_num):

            all_inputs_predict_correctly = True
            for x, label in zip(self.X, self.labels):
                predict_value = self.process(x)
                if predict_value == label:
                    continue
                else:
                    all_inputs_predict_correctly = False
                    self.weights += lr*(label - predict_value)* x
            print(str(epoch) + str(' : ') + str(self.weights[0]))

            if all_inputs_predict_correctly:
                return True


if __name__ == '__main__':
    data = pd.read_csv('./2.csv')
    train_data = [[int(x_1), int(x_2)] for x_1, x_2 in zip(data['x_1'], data['x_2'])]
    labels=[]
    for label in data['Label']:
        if label == 1:
            labels.append(int(label))
        else:
            labels.append(0)

    p = Peceptron(np.array(train_data),np.array(labels))
    if p.convergence_algorithm():
        print(p.weights)
        c_1 = []
        c_2 = []
        for i in range(len(labels)):
            if labels[i]==1:
                c_1.append(train_data[i])
            else:
                c_2.append(train_data[i])
        c_1 = np.array(c_1)
        c_2 = np.array(c_2)
        min_x = min(np.min(c_1[: , 0]) , np.min(c_2[: , 0]))
        max_x = max(np.max(c_1[:, 0]), np.max(c_2[:, 0]))
        plt.plot([min_x, max_x], [ (-p.weights[0][0] - p.weights[0][1] * min_x) / p.weights[0][2], (-p.weights[0][0] - p.weights[0][1] * max_x) / p.weights[0][2]])
        plt.scatter(c_1[:, 0], c_1[:, 1], c = 'r', marker = 'o')
        plt.scatter(c_2[:, 0], c_2[:, 1], c = 'b', marker = 'x')

        plt.show()
