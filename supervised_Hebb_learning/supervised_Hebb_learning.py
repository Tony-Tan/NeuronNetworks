import numpy as np
import os
import cv2


def symmetrical_hard_limit(x):
    """
    :param x:  input of hard limit transfer function, can be a vector or a scalar
    :return: result of hard limit transfer function
    """
    output=[]
    for i in x:
        if i>0:
            output.append(1)
        else:
            output.append(-1)
    return np.array(output)


class HebbLearning():

    def __init__(self, training_data_path='./data/train', gamma=0, alpha=0.5):
        """
        initial function
        :param training_data_path: the path of training set and their labels.
        They should be organized as pictures in '.png' form
        :param gamma: the punishment coefficients
        :param alpha: learning rate
        """
        self.gamma = gamma
        self.alpha = alpha
        x = self.load_data(training_data_path)
        self.X = np.array(x)
        self.label = np.array(x)
        self.weights = np.zeros((np.shape(x)[1], np.shape(x)[1]))
        self.test_data = []

    def load_data(self, data_path):
        """
        load image data and transfer it into matrix form
        :param data_path: the path of data
        :return: training set and targets respectively
        """
        name_list = os.listdir(data_path)
        X = []
        for file_name in name_list:
            data = cv2.imread(os.path.join(data_path, file_name), 0)
            if data is None:
                continue
            else:
                data = data.reshape(1, -1)[0].astype(np.float64)
            for i in range(len(data)):
                if data[i] > 0:
                    data[i] = 1
                else:
                    data[i] = -1
            data=data/np.linalg.norm(data)
            X.append(data)
        return X

    def process(self):
        """
        comput weights using Hebb learning function
        :return:
        """
        for x, label in zip(self.X, self.label):
            self.weights = self.weights + self.alpha * np.dot(label.reshape(-1,1), x.reshape(1,-1)) - self.gamma*self.weights

    def test(self, input_path='./data/test'):
        """
        test function used to test a given input use the linear associator
        :param input_path: test date should be organized as pictures whose names are its label
        :return: output label and
        """
        self.test_data = self.load_data(input_path)

        labels_test = []
        for x in self.test_data:
            output_origin = np.dot(self.weights,x.reshape(-1,1))
            labels_test.append(symmetrical_hard_limit(output_origin))
        return np.array(labels_test)


if __name__ == '__main__':
    h = HebbLearning(alpha=1.0)
    h.process()
    for i,x in zip(h.test(),h.test_data):
        image_repair = i.reshape(6,5).astype(np.uint8)
        image = x.reshape(6,5).astype(np.uint8)
        cv2.imshow('image_repaired',image_repair)
        cv2.imshow('image_org',image)
        cv2.waitKey(0)
        print(i)
