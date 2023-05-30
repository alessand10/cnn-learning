import numpy as np

class Perceptron1:
    def __init__(self, learning_rate = 0.01, random_number_seed = 1):
        random_number_generator = np.random.RandomState(random_number_seed)
        self.learning_rate = learning_rate
        self.random_number_seed = random_number_seed
        self.weights = random_number_generator.normal(loc=0.0, scale=0.01, size=2)
        self.bias = np.float_(0.)
        self.errors = []

    
    def net_input(self, x):
        return np.dot(x, self.weights) + self.bias
    
    def predict(self, x):
        weighted_output = self.net_input(x)
        return 1 if weighted_output > 0 else 0

    def fit(self, inputs, targets, epochs):

        for i in range(0, epochs):
            for input,target in zip(inputs, targets):
                prediction = self.predict(input)
                error = target - prediction
                self.weights += self.learning_rate * error * input
                self.bias += self.learning_rate * error


test_linear_eq = Perceptron1()

training_input = np.array([[1,1],[1,0],[0,1],[0,0]])
training_labels = np.array([1, 0, 0, 0])

test_linear_eq.fit(training_input,training_labels, 20)

print(test_linear_eq.predict([1,1]))


class Perceptron2:
    def __init__(self, learning_rate = 0.01, random_number_seed = 1):
        random_number_generator = np.random.RandomState(random_number_seed)
        self.learning_rate = learning_rate
        self.random_number_seed = random_number_seed
        self.weights = random_number_generator.normal(loc=0.0, scale=0.01, size=1)
        self.bias = np.float_(0.)
        self.errors = []

    
    def net_input(self, x):
        return np.dot(x, self.weights) + self.bias
    
    def predict(self, x):
        weighted_output = self.net_input(x)
        return weighted_output if weighted_output > 0 else 0

    def fit(self, inputs, targets, epochs):

        for i in range(0, epochs):
            for input,target in zip(inputs, targets):
                prediction = self.predict(input)
                error = target - prediction
                self.weights += self.learning_rate * error * input
                self.bias += self.learning_rate * error

test_linear_eq = Perceptron2()

training_input = np.array([1,2,3,4,5, 6, 7])
training_labels = np.array([2, 4, 6, 8, 10, 12, 14])

test_linear_eq.fit(training_input,training_labels, 50)

print(test_linear_eq.predict(8))