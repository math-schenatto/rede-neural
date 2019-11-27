import numpy as np
import random
from index import index

class Network():
    def __init__(self, neurons):
        self.neuron_layers = len(neurons)
        self.neurons = neurons
        self.biases = [np.random.uniform(low=0, high=1, size=(y, 1)) for y in neurons[1:]]
        self.weights = [np.random.uniform(low=0, high=1, size=(y,x)) for x, y in zip(neurons[:-1], neurons[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_inputs, epochs, learning_rate, test_inputs=None):
        if test_inputs:
            test_inputs = list(test_inputs)
            n_test = len(test_inputs)

        length = len(training_inputs)
        for epoch in range(epochs):
            packed_inputs = [training_inputs[k:k+1] for k in range(length)]

            for single_input in packed_inputs:
                self.update_input(single_input, learning_rate)

            if test_inputs:
                print("Epoch", epoch, ":", self.identify_many(test_inputs))
            else:
                print("Epoch", epoch)

    def update_input(self, single_input, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in single_input:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(learning_rate/len(single_input))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/len(single_input))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward
        activation = x
        # Lista para armazenar todas as ativações, camada por camada
        activations = [x]

        # lista para armazenar todos os vetores z, camada por camada
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.neuron_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return nabla_b, nabla_w

    def identify_many(self, training_inputs):
        return sum(int(self.identify(single_test)) for single_test in training_inputs)

    def identify(self, single_test, log=False):
        prediction = np.argmax(self.feedforward(single_test[0]))
        answer = np.argmax(single_test[1])
        if log:
            if prediction == answer:
                print("✅", index[answer], "→", index[prediction])
            else:
                print("❌", index[answer], "→", index[prediction])
        return prediction == answer

    # Retorna o vetor das derivadas parciais
    def cost_derivative(self, output_activations, y):
        return output_activations-y

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# Função para retornar as derivadas da função Sigmóide
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
