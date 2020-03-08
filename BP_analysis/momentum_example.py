import numpy as np
import matplotlib.pyplot as plt

def w(x):
    return 1.0+np.sin(2.0*np.pi*x/16.0)

def w_gamma(x, gamma=1.0):
    y=[]
    for i in range(len(x)):
        w_k = 1.0 + np.sin(2.0 * np.pi * x[i] / 16.0)
        if i == 0:
            y.append(0 * gamma + (1.0-gamma)*w_k)
        else:
            y.append(y[-1] * gamma + (1.0 - gamma) * w_k)
    return y


x = np.arange(0,200,1)
plt.figure(figsize=(16,8))
plt.plot(x, w_gamma(x, 0), c='r', lw=2, alpha=0.75, label = '$\gamma = 0.0$')
plt.plot(x, w_gamma(x, 0.9), c='g', lw=2, alpha=0.75, label = '$\gamma = 0.9$')
plt.plot(x, w_gamma(x, 0.98), c='b', lw=2, alpha=0.75, label = '$\gamma = 0.98$')
plt.legend(loc='upper right')
plt.show()