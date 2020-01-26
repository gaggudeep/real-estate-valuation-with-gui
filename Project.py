import numpy as np

class Perceptron:
    def __init__(self, learning_rate = 0.0001, epoch = 10**5):
        self.learning_rate = learning_rate
        self.epoch = epoch

    def training(self, inp, targets):
        self.weights = np.zeros(inp.shape[1])
        self.mse = []
        for i in range(self.epoch):
            self.gradient_descent(inp, targets)
            
    def predict(self, x):
        net_input = np.dot(x, self.weights)
        return net_input
    
    def gradient_descent(self, inp, targets):
        predictions = self.predict(inp)
        errors = predictions - targets
        self.mse.append(np.sum(errors**2))
        gradient = np.dot(inp.T, errors) / inp.shape[0]
        self.weights -= self.learning_rate * gradient

'''Importing data and visualizing'''
import pandas as pd
from matplotlib import pyplot as plt
data_df = pd.read_csv("dataset.csv")
del data_df["No"]
del data_df["Longitude"]
del data_df["Latitude"]
del data_df["Transaction date"]
corr_matrix = data_df.corr()
plt.figure(figsize = [10,10])
import seaborn as sb
sb.heatmap(corr_matrix)
targets = data_df["House price"].values
del data_df["House price"]
data = data_df.values

'''Normalizing feature set'''
data = (data - data.mean()) / data.std()

'''Data splitting'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size = 0.2)

'''Training model'''
p = Perceptron()
p.training(x_train, y_train)
print("\n\nWeights after training:-")
print(p.weights)

'''Testing model and visualizing results''' 
plt.figure(figsize = [15,4])
plt.title("ORIGINAL VALUES V/S PREDICTED VALUES")
plt.plot(np.arange(1, 84, 1), y_test, label = "Original", marker = 'o', linestyle = '')
plt.plot(np.arange(1, 84, 1), p.predict(x_test), label = "Prediction")
plt.legend()
plt.figure(figsize = [15,4])
plt.title("ITERATION V/S MEAN SQUARE ERROR")
plt.plot(np.arange(1, (10**5 + 1), 1), p.mse)