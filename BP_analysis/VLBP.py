import numpy as np
import matplotlib.pyplot as plt



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
        self.delta_weights_array = []
        self.delta_weights_k_1 = np.zeros(num_of_input_connections+1)
        self.weights_backup = np.zeros(num_of_input_connections + 1)
        self.delta_weights = np.zeros(num_of_input_connections + 1)
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
            self.delta_weights_array.append(-self.sensitivity * inputs)

    def undo_update_weights_process(self):
        if not self.input_layer_flag:
            self.weights = self.weights_backup
            self.sensitivity = 1

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

    def cal_delta_weights(self):
        forward_bfs_list = self.next_neurons
        for n in forward_bfs_list:
            n.cal_delta_weights()
        if not self.input_layer_flag and len(self.delta_weights_array)!=0:
            self.delta_weights = np.sum(self.delta_weights_array, axis=0) / (np.size(self.delta_weights_array)) * len(self.weights)
            self.delta_weights_array = []

    def update_weights(self, lr, gamma):
        forward_bfs_list = self.next_neurons
        for n in forward_bfs_list:
            n.update_weights(lr, gamma)
        if not self.input_layer_flag and self.sensitivity!=0:
            self.weights_backup = self.weights
            self.weights = self.weights + (gamma*self.delta_weights_k_1 + (1-gamma)*self.delta_weights)*lr
            self.delta_weights_k_1 = self.delta_weights
            self.sensitivity = 0

    def undo_update_weights(self):
        forward_bfs_list = self.next_neurons
        for n in forward_bfs_list:
            n.undo_update_weights()
        self.undo_update_weights_process()



class Net():
    def __init__(self, neurons, perfromace_index, learning_rate,gamma):
        self.last_neuron = neurons
        self.start_point = neurons.start_point
        self.output = 0
        self.p_i = perfromace_index
        self.loss = 0
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gamma_original = gamma

    def forward(self, input):
        self.start_point.output = input
        self.last_neuron.forward()
        self.output = self.last_neuron.output
        return self.output

    def train(self, inputs, targets, method = 'SDBP', xi = 0.04, rou = 0.7, eta= 1.05):
        if self.loss == 0:
            self.loss = self.p_i.c(self.forward(inputs),targets)
        for p,t in zip(inputs,targets):
            self.start_point.output = p
            self.last_neuron.forward()
            self.output = self.last_neuron.output
            self.start_point.backward(self.p_i.derivative(self.output, t))

        self.start_point.cal_delta_weights()
        if method == 'VLBP':
            self.start_point.update_weights(self.learning_rate,self.gamma)
            new_loss = self.p_i.c(self.forward(inputs),targets)

            if new_loss/self.loss > 1.+xi:
                while new_loss/self.loss > 1.+xi:
                    self.start_point.undo_update_weights()
                    self.gamma = 0.0
                    self.learning_rate = self.learning_rate * rou
                    self.start_point.update_weights(self.learning_rate, self.gamma)
                    new_loss = self.p_i.c(self.forward(inputs), targets)
            elif new_loss/self.loss <=1:
                self.learning_rate = self.learning_rate*eta
                if self.gamma == 0:
                    self.gamma = self.gamma_original
            else:
                if self.gamma == 0:
                    self.gamma = self.gamma_original

            self.loss = new_loss
        else:
            self.start_point.update_weights(self.learning_rate, self.gamma)


class MSE():
    def c(self, a, t):
        return np.dot((a - t).transpose(),(a - t))

    def derivative(self, a, t):
        return -2*(t-a)


# 12-2 example
def generating_training_set():
    x = []
    t = []
    logsig = log_sigmoid()
    for i in np.arange(-2,2.1,0.1):
        x.append(i)
        t.append(logsig.f(logsig.f(10*i-5)*1+logsig.f(10*i+5)*1-1))
    return np.array(x),np.array(t)

def mse_example_w(X,Y, p, t,size_x,size_y, b_11,w_12,b_12, w_22,b_2):
    logsig = log_sigmoid()
    sum = np.zeros([size_x,size_y])
    for p_i,t_i in zip(p,t):
        a_11 = logsig.f(X * p_i + b_11)
        a_12 = logsig.f(w_12 * p_i + b_12)
        a_21 = logsig.f(a_11 * Y + a_12*w_22 + b_2)
        sum = sum + (a_21-t_i)*(a_21-t_i)
    return sum/np.size(p)

def mse_example_b(X,Y, p, t,size_x,size_y, w_11,w_12, w_21,w_22,b_2):
    logsig = log_sigmoid()
    sum = np.zeros([size_x,size_y])
    for p_i,t_i in zip(p,t):
        a_11 = logsig.f(w_11 * p_i + X)
        a_12 = logsig.f(w_12 * p_i + Y)
        a_21 = logsig.f(a_11 * w_21 + a_12*w_22 + b_2)
        sum = sum + (a_21-t_i)*(a_21-t_i)
    return sum/np.size(p)

if __name__ == '__main__':
    X, T = generating_training_set()

    lr = 100
    w_1_11 = -4
    w_2_11 = -4
    n_1 = Neuron(inputs=0)
    n_2_1 = Neuron(num_of_input_connections=1, transfer_function=log_sigmoid(), previous_neurons=[n_1])
    n_2_1.initial_weights([w_1_11,-5])
    n_2_2 = Neuron(num_of_input_connections=1, transfer_function=log_sigmoid(), previous_neurons=[n_1])
    n_2_2.initial_weights([10, 5])
    n_3_1 = Neuron(num_of_input_connections=2, transfer_function=log_sigmoid(), previous_neurons=[n_2_1, n_2_2])
    n_3_1.initial_weights([w_2_11, 1, -1])
    loss_function = MSE()
    net = Net(n_3_1, loss_function,lr,0)
    last_x = []
    last_y = []
    train_SDBP = True

    '''VLBP'''
    n_1_VLBP = Neuron(inputs=0)
    n_2_1_VLBP = Neuron(num_of_input_connections=1, transfer_function=log_sigmoid(), previous_neurons=[n_1_VLBP])
    n_2_1_VLBP.initial_weights([w_1_11, -5])
    n_2_2_VLBP = Neuron(num_of_input_connections=1, transfer_function=log_sigmoid(), previous_neurons=[n_1_VLBP])
    n_2_2_VLBP.initial_weights([10, 5])
    n_3_1_VLBP = Neuron(num_of_input_connections=2, transfer_function=log_sigmoid(), previous_neurons=[n_2_1_VLBP, n_2_2_VLBP])
    n_3_1_VLBP.initial_weights([w_2_11, 1, -1])
    loss_function_VLBP = MSE()
    net_VLBP = Net(n_3_1_VLBP, loss_function_VLBP,lr,0)

    last_x_VLBP = []
    last_y_VLBP = []
    train_VLBP = True

    '''MOBP'''
    n_1_MOBP = Neuron(inputs=0)
    n_2_1_MOBP = Neuron(num_of_input_connections=1, transfer_function=log_sigmoid(), previous_neurons=[n_1_MOBP])
    n_2_1_MOBP.initial_weights([w_1_11, -5])
    n_2_2_MOBP = Neuron(num_of_input_connections=1, transfer_function=log_sigmoid(), previous_neurons=[n_1_MOBP])
    n_2_2_MOBP.initial_weights([10, 5])
    n_3_1_MOBP = Neuron(num_of_input_connections=2, transfer_function=log_sigmoid(),
                        previous_neurons=[n_2_1_MOBP, n_2_2_MOBP])
    n_3_1_MOBP.initial_weights([w_2_11, 1, -1])
    loss_function_MOBP = MSE()
    net_MOBP = Net(n_3_1_MOBP, loss_function_MOBP, lr, 0.2)

    last_x_MOBP = []
    last_y_MOBP = []
    train_MOBP = True
    for epoch in range(10000):
        # SDBP
        n_2_1.initial_weights([n_2_1.weights[0], -5])
        n_2_2.initial_weights([10, 5])
        n_3_1.initial_weights([n_3_1.weights[0], 1, -1])

        # VLBP
        n_2_1_VLBP.initial_weights([n_2_1_VLBP.weights[0], -5])
        n_2_2_VLBP.initial_weights([10, 5])
        n_3_1_VLBP.initial_weights([n_3_1_VLBP.weights[0], 1, -1])

        # MOBP
        n_2_1_MOBP.initial_weights([n_2_1_MOBP.weights[0], -5])
        n_2_2_MOBP.initial_weights([10, 5])
        n_3_1_MOBP.initial_weights([n_3_1_MOBP.weights[0], 1, -1])
        if epoch%100==0:
            n = 200
            x = np.linspace(-5, 15, n)
            y = np.linspace(-5, 15, n)

            X_c, Y_c = np.meshgrid(x, y)
            plt.figure(figsize=(8, 8))
            Z = [0.001, 0.002, 0.004, 0.006, 0.008, 0.01,0.012,0.014,0.016,0.018,0.02,0.03]
            C = plt.contour(X_c, Y_c, mse_example_w(X_c, Y_c, X, T, n, n, n_2_1.weights[1],
                                                  n_2_2.weights[0], n_2_2.weights[1],n_3_1.weights[1],
                                                  n_3_1.weights[2]),
                            Z, alpha=0.7, cmap=plt.cm.rainbow)
            plt.clabel(C, inline=True, fontsize=10)
            plt.xlabel('$W^1_{11}$')
            plt.ylabel('$W^2_{11}$')

            # SDBP
            e = loss_function.c(net.forward(X), T)
            print('epoch ' + str(epoch) + ' loss = ' + str(e)+ ' | learning rate: '+str(net.learning_rate))

            # VLBP
            e_VLBP = loss_function.c(net_VLBP.forward(X), T)
            print('epoch ' + str(epoch) + ' loss_VLBP = ' + str(e_VLBP)+ ' | learning rate: '+str(net_VLBP.learning_rate))

            # MOBP
            e_MOBP = loss_function.c(net_MOBP.forward(X), T)
            print('epoch ' + str(epoch) + ' loss_MOBP = ' + str(e_MOBP) + ' | learning rate: ' + str(
                net_MOBP.learning_rate))

            if e < 0.0001:
                train_SDBP = False
            if e_VLBP < 0.0001:
                train_VLBP = False
            if e_MOBP < 0.0001:
                train_MOBP = False

            # SDBP
            last_x.append(n_2_1.weights[0])
            last_y.append(n_3_1.weights[0])
            plt.scatter(last_x, last_y, c='b', marker='x', alpha=0.7)
            plt.plot(last_x, last_y, c='b', linestyle='-.', linewidth=2, alpha=0.7,label='trajectory of SDBP')

            # VLBP
            last_x_VLBP.append(n_2_1_VLBP.weights[0])
            last_y_VLBP.append(n_3_1_VLBP.weights[0])
            plt.scatter(last_x_VLBP, last_y_VLBP, c='r', marker='*', alpha=0.7)
            plt.plot(last_x_VLBP,last_y_VLBP,c='r',linestyle='-.',linewidth=2,alpha=0.7, label='trajectory of VLBP')

            # MOBP
            last_x_MOBP.append(n_2_1_MOBP.weights[0])
            last_y_MOBP.append(n_3_1_MOBP.weights[0])
            plt.scatter(last_x_MOBP, last_y_MOBP, c='g', marker='+', alpha=0.7)
            plt.plot(last_x_MOBP, last_y_MOBP, c='g', linestyle='-.', linewidth=2, alpha=0.7,
                     label='trajectory of MOBP')

            plt.legend(loc='upper right')
            plt.savefig('./data/contour_plot_' + str(epoch) + '.png')
            plt.close()

        if train_SDBP:
            net.train(X, T)
        if train_VLBP:
            net_VLBP.train(X, T, method='VLBP')
        if train_MOBP:
            net_MOBP.train(X, T)
        if not train_SDBP and not train_VLBP and not train_MOBP:
            break