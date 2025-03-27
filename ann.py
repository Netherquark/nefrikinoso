import csv
import random
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import itertools

class Dataset:
    def __init__(self, filename):
        self.filename = filename
        self.data = []
        self.minmax = []
        self.load_csv()
        self.clean_data()
        self.convert_to_float()
        self.convert_to_int()
        self.normalize()

    def load_csv(self):
        df = pd.read_csv(self.filename, encoding='utf-8-sig')
        df.dropna(inplace=True)
        df['grf'] = pd.to_numeric(df['grf'], errors='coerce')
        df.dropna(inplace=True)
        df['class'] = df['class'].map({'ckd': 1, 'notckd': 0})
        self.data = df.values.tolist()

    def clean_data(self):
        for row in self.data:
            for i in range(len(row)):
                if isinstance(row[i], str):
                    row[i] = row[i].strip().replace('?', '0')

    def convert_to_float(self):
        for row in self.data:
            for i in range(len(row) - 1):
                try:
                    row[i] = float(row[i])
                except ValueError:
                    row[i] = 0.0

    def convert_to_int(self):
        class_values = [row[-1] for row in self.data]
        unique_values = set(class_values)
        lookup = {value: i for i, value in enumerate(unique_values)}
        for row in self.data:
            row[-1] = lookup[row[-1]]

    def normalize(self):
        self.minmax = [[min(col), max(col)] for col in zip(*self.data)]
        for row in self.data:
            for i in range(len(row) - 1):
                if self.minmax[i][1] != self.minmax[i][0]:
                    row[i] = (row[i] - self.minmax[i][0]) / (self.minmax[i][1] - self.minmax[i][0])
                else:
                    row[i] = 0.0

class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden=10, n_outputs=1, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.network = self.initialize_network(n_inputs, n_hidden, n_outputs)

    def initialize_network(self, n_inputs, n_hidden, n_outputs):
        hidden_layer = [{'weights': np.random.randn(n_inputs + 1) * 0.1} for _ in range(n_hidden)]
        output_layer = [{'weights': np.random.randn(n_hidden + 1) * 0.1} for _ in range(n_outputs)]
        return [hidden_layer, output_layer]

    def activate(self, weights, inputs):
        return np.dot(weights[:-1], inputs) + weights[-1]

    def transfer(self, activation):
        return 1.0 / (1.0 + np.exp(-activation))

    def transfer_derivative(self, output):
        return output * (1.0 - output)

    def forward_propagate(self, row):
        inputs = row[:-1]
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = []
            if i != len(self.network) - 1:
                for j in range(len(layer)):
                    error = sum(neuron['weights'][j] * neuron['delta'] for neuron in self.network[i + 1])
                    errors.append(error)
            else:
                errors = [(expected[j] - neuron['output']) for j, neuron in enumerate(layer)]
            for j, neuron in enumerate(layer):
                neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output'])

    def update_weights(self, row):
        for i, layer in enumerate(self.network):
            inputs = row[:-1] if i == 0 else [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in layer:
                for j in range(len(inputs)):
                    neuron['weights'][j] += self.learning_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += self.learning_rate * neuron['delta']

    def train(self, train_data, n_outputs):
        for _ in range(self.epochs):
            for row in train_data:
                outputs = self.forward_propagate(row)
                expected = [0] * max(2, n_outputs)
                if 0 <= int(row[-1]) < len(expected):
                    expected[int(row[-1])] = 1
                self.backward_propagate_error(expected)
                self.update_weights(row)

    def predict(self, row):
        outputs = self.forward_propagate(row)
        return 1 if outputs[0] > 0.5 else 0

class CrossValidator:
    @staticmethod
    def accuracy_metric(actual, predicted):
        return sum(1 for a, p in zip(actual, predicted) if a == p) / len(actual) * 100.0

    def evaluate(self, dataset, model):
        train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
        model.train(train_data, 1)
        predictions = [model.predict(row) for row in test_data]
        actual = [row[-1] for row in test_data]
        return self.accuracy_metric(actual, predictions)

def hyperparameter_tuning(dataset):
    param_grid = {
        'n_hidden': [8, 16, 32],
        'learning_rate': [0.01, 0.005, 0.001],
        'epochs': [1000, 2000, 5000]
    }
    
    best_accuracy = 0
    best_params = None
    
    for n_hidden, learning_rate, epochs in itertools.product(param_grid['n_hidden'], param_grid['learning_rate'], param_grid['epochs']):
        model = NeuralNetwork(n_inputs=len(dataset.data[0]) - 1, n_hidden=n_hidden, epochs=epochs, learning_rate=learning_rate)
        validator = CrossValidator()
        accuracy = validator.evaluate(dataset.data, model)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = (n_hidden, learning_rate, epochs)
        
        print(f'Params: Hidden={n_hidden}, LR={learning_rate}, Epochs={epochs} --> Accuracy: {accuracy:.2f}%')
    
    print(f'Best Params: Hidden={best_params[0]}, LR={best_params[1]}, Epochs={best_params[2]} --> Best Accuracy: {best_accuracy:.2f}%')

if __name__ == "__main__":
    dataset = Dataset("/home/r1ddh1/2nd_year/pbl_sem4/nefrikinoso/ckd_prediction_dataset.csv")
    hyperparameter_tuning(dataset)
