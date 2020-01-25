import numpy as np

class Perceptron:
    def __init__(self, learning_rate = 0.01, epoch = 100):
        self.learning_rate = learning_rate
        self.epoch = epoch

    def training(self, input, target):
        self.weights = np.zeros(input.shape[1] + 1)
        for i in range(self.epoch):
            for xi, tar in zip(input, target):
                output = self.predict(xi)
                error = tar - output
                if (error != 0):
                    update = self.learning_rate * error
                    self.weights[0] += update
                    self.weights[1:] += update * xi
    
    def predict(self, xi):
        net_input = np.dot(xi, self.weights[1:]) + self.weights[0]
        return 1 if (net_input > 0) else 0

'''Importing data'''
import pandas as pd
data_df = pd.read_csv("dataset.csv")
del data_df["No"]
del data_df["Transaction date"]
target = np.zeros(data_df.shape[0])
house_prices = data_df["House price"].values
for i in range(target.shape[0]):
    if house_prices[i] > 38:
        target[i] = 1
del data_df["House price"]
data = data_df.values

'''Data splitting'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.2)

'''Training model'''
p = Perceptron()
p.training(x_train, y_train)
print("Weights after training:-")
print(p.weights)

'''Testing model and visualizing results''' 
from matplotlib import pyplot as plt
figure = plt.figure(figsize=(15, 6))
ax = figure.add_subplot(111)
ax.plot(np.arange(0, 83), y_test, label = 'Original', marker = '.')
y = np.empty((x_test.shape[0]))
for i in range(x_test.shape[0]):
    y[i] = p.predict(x_test[i])
ax.plot(np.arange(0, 83), y, label = 'Test', marker = '.')
plt.legend(loc = 1)
plt.show()
print('Accuracy:', np.sum(y == y_test) * 100 / y_test.shape[0])
