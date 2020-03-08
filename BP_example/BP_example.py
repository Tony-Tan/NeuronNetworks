import numpy as np
import matplotlib.pyplot as plt
lr=0.1


class log_sigmoid():
    def f(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def d(self, x):
        return (1-self.f(x))*self.f(x)


class linear():
    def f(self, x):
        return x

    def d(self, x):
        return 1


class Neuron():
    def __init__(self, inputs = None, num_of_input_connections = 0, transfer_function = None, previous_neurons = None):
        self.next_neurons = []
        self.previous_neurons = []
        self.start_point = self
        self.delta_update = []
        if inputs is not None:
            self.output = inputs
            self.input_layer_flag = True
        else:
            self.num_of_input_connections = num_of_input_connections
            self.transfer_function_f = transfer_function.f
            self.transfer_function_d = transfer_function.d
            self.sensitivity = 0
            self.previous_neurons = previous_neurons
            self.output = 0.0
            self.weights = None
            self.initial_weights(weights=None)
            self.start_point = previous_neurons[0].start_point
            self.input_layer_flag = False
            self.n = 0
            for p_n in self.previous_neurons:
                p_n.next_neurons.append(self)

    def initial_weights(self, weights=None):
        if weights is not None:
            self.weights = np.array(weights)
        else:
            # initial weights and bias with random number in [-1,1]
            self.weights = np.random.uniform(-10.1, 10.1, size=self.num_of_input_connections + 1)

    def forward_process(self):
        if not self.input_layer_flag:
            inputs = []
            for p in self.previous_neurons:
                inputs.append(p.output)
            inputs.append(1)
            inputs = np.array(inputs)
            self.n = np.dot(inputs, self.weights.transpose())
            self.output = self.transfer_function_f(self.n)

    def backward_process(self, performance_index_d):
        if len(self.next_neurons) == 0:
            self.sensitivity = performance_index_d*self.transfer_function_d(self.n)
        elif not self.input_layer_flag:
            n = []
            w = []
            s = []
            for n_n in self.next_neurons:
                w.append(n_n.weights[n_n.previous_neurons.index(self)])
                s.append([n_n.sensitivity])
            n = np.array(self.n)
            w = np.array(w)
            s = np.array(s)
            f_n = self.transfer_function_d(n)
            self.sensitivity=np.dot(np.dot(f_n,w),s)
        if not self.input_layer_flag:
            inputs = []
            for p in self.previous_neurons:
                inputs.append(p.output)
            inputs.append(1)
            inputs = np.array(inputs)
            self.delta_update.append(self.sensitivity * inputs)

    def update_weights_process(self):
        if not self.input_layer_flag and len(self.delta_update) != 0:
            self.weights = self.weights - np.sum(self.delta_update,axis = 0)/(np.size(self.delta_update)) * lr * len(self.weights)
            self.sensitivity = None
            self.delta_update = []

    def forward(self):
        backward_bfs_list = self.previous_neurons
        for n in backward_bfs_list:
            n.forward()
        self.forward_process()

    def backward(self, performance_index_d):
        forward_bfs_list = self.next_neurons
        for n in forward_bfs_list:
            n.backward(performance_index_d)
        self.backward_process(performance_index_d)

    def update_weight(self):
        forward_bfs_list = self.next_neurons
        for n in forward_bfs_list:
            n.update_weight()
        self.update_weights_process()


class Net():
    def __init__(self, neurons, perfromace_index):
        self.last_neuron = neurons
        self.start_point = neurons.start_point
        self.output = 0
        self.p_i = perfromace_index
        self.loss = 0

    def forward(self, input):
        self.start_point.output = input
        self.last_neuron.forward()
        self.output = self.last_neuron.output
        return self.output

    def train(self, inputs, targets):
        for p,t in zip(inputs,targets):
            self.start_point.output = p
            self.last_neuron.forward()
            self.output = self.last_neuron.output
            self.start_point.backward(self.p_i.derivative(self.output, t))
        self.start_point.update_weight()


class MSE():
    def c(self, a, t):
        return np.dot((a - t).transpose(),(a - t))

    def derivative(self, a, t):
        return -2*(t-a)


# 11-14 example
def generating_training_set():
    x = []
    t = []
    for i in np.arange(-2,2.1,0.4):
        x.append(i)
        t.append(1+np.sin(np.pi/4.0*i))
    return np.array(x),np.array(t)


if __name__ == '__main__':
    X, T = generating_training_set()
    n_1 = Neuron(inputs=0)
    n_2_1 = Neuron(num_of_input_connections=1, transfer_function=log_sigmoid(), previous_neurons=[n_1])
    n_2_2 = Neuron(num_of_input_connections=1, transfer_function=log_sigmoid(), previous_neurons=[n_1])
    n_3_1 = Neuron(num_of_input_connections=2, transfer_function=linear(), previous_neurons=[n_2_1, n_2_2])

    loss_function = MSE()
    net = Net(n_3_1, loss_function)
    plt.ion()
    plt.figure(figsize=(8, 8))
    plt.title('$i=1$')
    x_axis = np.arange(-2,2,0.001)
    for epoch in range(200000):
        if epoch % 1000 == 0:
            e = loss_function.c(net.forward(X), T )/np.size(X)
            if e < 0.0000001:
                break
            print('epoch '+str(epoch) + ' loss = '+str(e))
            plt.cla()
            outputs = []
            for x in X:
                net.forward(x)
                outputs.append(net.output)
            plt.scatter(X, outputs, c='b', alpha=0.4, lw=1, label='BP output')
            plt.scatter(X, T, c='r', alpha=0.4, lw=1, label='Sample Points')
            outputs = []
            for x in x_axis:
                net.forward(x)
                outputs.append(net.output)
            plt.plot(x_axis, outputs, c='b', alpha=0.8, lw=1, label='BP output Plot')
            plt.plot(X, 1+np.sin(np.pi/4.0*X), c='r', alpha=0.8, lw=1, label='Ground Truth')
            plt.legend(loc='upper left')
            plt.savefig('./data/' + str(epoch) + '.png')
            print('./data/' + str(epoch) + '.png has been saved')
            print(n_2_1.weights)
            print(n_2_2.weights)
            print(n_3_1.weights)
        net.train(X,T)