import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
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
    return (np.exp(x)-np.exp(-x)) / (np.exp(x) + np.exp(-x))


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
    def __init__(self, x, labels, transfer_function_type = 'hard_limit', weight_initial_type = 'ordered' ):
        self.input_bias_padding = np.ones([ np.shape(x)[0],1])
        self.X = np.c_[self.input_bias_padding, x]
        self.labels = labels
        self.modification_times = 0
        self.epoch_times = 0
        if weight_initial_type == 'ordered':
            self.weights = np.zeros([1,np.shape(self.X)[1]])
        elif weight_initial_type == 'random_p':
            self.weights = [self.X[random.randint(0,len(x)-1)]]
        elif weight_initial_type == 'average_p':
            tempor_sum=np.zeros([1,np.shape(self.X)[1]])
            for i in range(len(x)):
                tempor_sum+=[self.X[i]]
            self.weights = tempor_sum/len(x)
        elif weight_initial_type == 'random_w':
            self.weights = [[random.randint(-1000,1000),random.randint(-1000,1000),random.randint(-1000,1000)]]
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
            random_zip = list(zip(self.X, self.labels))
            random.shuffle(random_zip)
            for x_label in random_zip:
                x = x_label[0]
                label = x_label[1]
                predict_value = self.process(x)
                if predict_value > 0:
                    predict_value = 1
                else:
                    predict_value = 0
                if predict_value == label:
                    continue
                else:
                    all_inputs_predict_correctly = False
                    self.weights += lr * (label - predict_value) * x
                    self.modification_times += 1
            if all_inputs_predict_correctly:
                self.epoch_times = epoch
                return True
            #print(str(epoch) + '-' + str(num_of_modification) + str(' : ') + str(self.weights[0]))


if __name__ == '__main__':
    data = pd.read_csv('./4.csv')
    train_data = [[int(x_1), int(x_2)] for x_1, x_2 in zip(data['x_1'], data['x_2'])]
    labels=[]
    for label in data['Label']:
        if label == 1:
            labels.append(int(label))
        else:
            labels.append(0)

    statistic_of_random_sequence = []
    statistic_of_original_sequence = []
    time_axis=[]
    for i in range(100):
        print(str(i) +'times')
        time_axis.append(i)
        p = Peceptron(np.array(train_data), np.array(labels), 'hard_limit', weight_initial_type='ordered')
        if p.convergence_algorithm():
            statistic_of_random_sequence.append([p.epoch_times, p.modification_times])

        statistic_of_original_sequence.append([55448, 188866])
    plt.rcParams['figure.figsize'] = (16.0, 4.0)
    plt.figure(1)
    plt.plot(time_axis, np.array(statistic_of_random_sequence)[:, 0], 'C1--',
             label='Random Sequence'+'(Mean:%.3f'%np.mean(np.array(statistic_of_random_sequence), axis=0)[0]+')', linewidth=1)
    plt.plot(time_axis, np.array(statistic_of_original_sequence)[:, 0], 'C2--',
             label='Original Ordered Sequence'+'(Mean:%.3f'%np.mean(np.array(statistic_of_original_sequence), axis=0)[0]+')', linewidth=1)
    plt.legend(loc='upper right')
    plt.savefig('training_set_order_comparison_epoch.png', dpi=300)
    plt.figure(2)
    plt.plot(time_axis, np.array(statistic_of_random_sequence)[:, 1], 'C1--',
             label='Random Sequence'+'(Mean:%.3f'%np.mean(np.array(statistic_of_random_sequence), axis=0)[1]+')', linewidth=1)
    plt.plot(time_axis, np.array(statistic_of_original_sequence)[:, 1], 'C2--',
             label='Original Ordered Sequence'+'(Mean:%.3f'%np.mean(np.array(statistic_of_original_sequence), axis=0)[1]+')', linewidth=1)
    plt.legend(loc='upper right')
    plt.savefig('training_set_order_comparison_modify.png', dpi=300)
    plt.show()

